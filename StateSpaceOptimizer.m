classdef StateSpaceOptimizer < handle
    properties
        model
        preprocessor
        learn_rate = 0.001
        mini_batch_size = 32
        max_epochs = 50
        early_stop_limit = 5
        
        best_valid_loss = Inf
        stop_counter = 0
        train_loss_history = []
        valid_loss_history = []
        
        tag_metrics = struct('precision', [], 'recall', [], 'f1', [])
    end
    
    methods
        function obj = StateSpaceOptimizer(model, preprocessor)
            obj.model = model;
            obj.preprocessor = preprocessor;
        end
        
        function optimize(obj, train_inputs, train_outputs, valid_inputs, valid_outputs)
            for epoch = 1:obj.max_epochs
                epoch_loss = 0;
                all_train_preds = [];
                all_train_true = [];
                
                num_batches = floor(length(train_inputs) / obj.mini_batch_size);
                for batch = 1:num_batches
                    start_idx = (batch-1) * obj.mini_batch_size + 1;
                    end_idx = min(batch * obj.mini_batch_size, length(train_inputs));
                    
                    batch_inputs = train_inputs(start_idx:end_idx);
                    batch_outputs = train_outputs(start_idx:end_idx);
                    
                    [batch_loss, batch_preds, batch_true] = obj.optimize_batch(batch_inputs, batch_outputs);
                    epoch_loss = epoch_loss + batch_loss;
                    
                    all_train_preds = [all_train_preds; batch_preds];
                    all_train_true = [all_train_true; batch_true];
                end
                
                epoch_loss = epoch_loss / num_batches;
                metrics = obj.compute_metrics(all_train_preds, all_train_true);
                
                fprintf('\nEpoch %d/%d:\n', epoch, obj.max_epochs);
                fprintf('Epoch Loss: %.4f | Accuracy: %.2f%%\n\n', epoch_loss, metrics.accuracy);
                
                fprintf('Class    Precision    Recall      Accuracy\n');
                fprintf('------------------------------------\n');
                
                tag_names = {'Noun', 'Verb', 'Adj/Adv', 'Other'};
                for i = 1:length(tag_names)
                    fprintf('%-8s %-12.2f%% %-12.2f%% %-12.2f%%\n', ...
                        tag_names{i}, ...
                        metrics.precision(i), ...
                        metrics.recall(i), ...
                        metrics.accuracy_per_class(i));
                end
                
                if obj.check_early_stop(epoch_loss)
                    fprintf('\nEarly stopping triggered!\n');
                    break;
                end
            end
        end
        
        
        
        function [loss, batch_predictions, batch_true_labels] = optimize_batch(obj, batch_inputs, batch_outputs)
            total_loss = 0;
            all_predictions = [];
            all_true_labels = [];
            gradients = struct();
            
            gradients.kernel = zeros(size(obj.model.kernel));
            gradients.disc_residual = zeros(size(obj.model.disc_residual));
            gradients.class_weights = zeros(size(obj.model.class_weights));
            gradients.class_bias = zeros(size(obj.model.class_bias));
            
            for i = 1:length(batch_inputs)
                sequence = batch_inputs{i};
                labels = batch_outputs{i};
                
                [windows, num_windows] = obj.model.process_sequence(sequence);
                
                seq_predictions = zeros(num_windows, obj.model.output_dim);
                seq_hidden_states = zeros(num_windows, obj.model.hidden_dim);
                
                for w = 1:num_windows
                    current_window = windows(w,:,:);
                    current_window = reshape(current_window, [1, obj.model.context_size, obj.model.hidden_dim]);
                    
                    [predictions, hidden_state] = obj.model.forward(current_window);
                    seq_predictions(w,:) = predictions;
                    seq_hidden_states(w,:) = hidden_state;
                end
                
                one_hot_labels = zeros(num_windows, obj.model.output_dim);
                for w = 1:num_windows
                    one_hot_labels(w, labels(w)) = 1;
                end
                
                epsilon = 1e-15;
                seq_predictions = max(min(seq_predictions, 1-epsilon), epsilon);
                seq_loss = -sum(sum(one_hot_labels .* log(seq_predictions)));
                total_loss = total_loss + seq_loss;
                
                d_predictions = -(one_hot_labels ./ seq_predictions);
                d_logits = seq_predictions .* (d_predictions - sum(d_predictions .* seq_predictions, 2));
                
                d_class_weights = d_logits' * seq_hidden_states;
                d_class_bias = sum(d_logits, 1)';
                
                d_hidden = d_logits * obj.model.class_weights;
                
                for w = 1:num_windows
                    x = squeeze(windows(w,:,:));
                    d_h = d_hidden(w,:);
                    
                    for t = 1:obj.model.context_size
                        for j = 1:t
                            gradients.kernel(j) = gradients.kernel(j) + d_h * x(t,:)';
                        end
                    end
                    
                    for t = 1:obj.model.context_size
                        gradients.disc_residual = gradients.disc_residual + d_h' * x(t,:);
                    end
                end
                
                gradients.class_weights = gradients.class_weights + d_class_weights;
                gradients.class_bias = gradients.class_bias + d_class_bias;
                
                [~, pred_classes] = max(seq_predictions, [], 2);
                all_predictions = [all_predictions; pred_classes];
                all_true_labels = [all_true_labels; labels'];
            end
            
            batch_size = length(batch_inputs);
            total_loss = total_loss / batch_size;
            gradients.kernel = gradients.kernel / batch_size;
            gradients.disc_residual = gradients.disc_residual / batch_size;
            gradients.class_weights = gradients.class_weights / batch_size;
            gradients.class_bias = gradients.class_bias / batch_size;
            
            obj.model.kernel = obj.model.kernel - obj.learn_rate * gradients.kernel;
            obj.model.disc_residual = obj.model.disc_residual - obj.learn_rate * gradients.disc_residual;
            obj.model.class_weights = obj.model.class_weights - obj.learn_rate * gradients.class_weights;
            obj.model.class_bias = obj.model.class_bias - obj.learn_rate * gradients.class_bias;
            
            loss = total_loss;
            batch_predictions = all_predictions;
            batch_true_labels = all_true_labels;
        end
        
        function [val_loss, all_predictions, all_true_labels] = evaluate(obj, val_inputs, val_outputs)
            total_loss = 0;
            all_predictions = [];
            all_true_labels = [];
            
            for i = 1:length(val_inputs)
                sequence = val_inputs{i};
                labels = val_outputs{i};
                
                [windows, num_windows] = obj.model.process_sequence(sequence);
                
                seq_predictions = zeros(num_windows, obj.model.output_dim);
                
                for w = 1:num_windows
                    current_window = windows(w,:,:);
                    current_window = reshape(current_window, [1, obj.model.context_size, obj.model.hidden_dim]);
                    
                    predictions = obj.model.forward(current_window);
                    seq_predictions(w,:) = predictions;
                end
                
                one_hot_labels = zeros(num_windows, obj.model.output_dim);
                for w = 1:num_windows
                    one_hot_labels(w, labels(w)) = 1;
                end
                
                epsilon = 1e-15;
                seq_predictions = max(min(seq_predictions, 1-epsilon), epsilon);
                seq_loss = -sum(sum(one_hot_labels .* log(seq_predictions)));
                
                total_loss = total_loss + seq_loss;
                [~, pred_classes] = max(seq_predictions, [], 2);
                all_predictions = [all_predictions; pred_classes];
                all_true_labels = [all_true_labels; labels'];
            end
            
            val_loss = total_loss / length(val_inputs);
        end
        
        function metrics = compute_metrics(obj, predictions, true_labels)
            metrics = struct();
            
            metrics.accuracy = mean(predictions == true_labels) * 100;
            
            num_classes = obj.preprocessor.output_dim;
            metrics.precision = zeros(num_classes, 1);
            metrics.recall = zeros(num_classes, 1);
            metrics.accuracy_per_class = zeros(num_classes, 1);
            
            for class = 1:num_classes
                tp = sum((predictions == class) & (true_labels == class));
                fp = sum((predictions == class) & (true_labels ~= class));
                fn = sum((predictions ~= class) & (true_labels == class));
                tn = sum((predictions ~= class) & (true_labels ~= class));
                
                if tp + fp == 0
                    metrics.precision(class) = 0;
                else
                    metrics.precision(class) = tp / (tp + fp) * 100;
                end
                
                if tp + fn == 0
                    metrics.recall(class) = 0;
                else
                    metrics.recall(class) = tp / (tp + fn) * 100;
                end
                
                metrics.accuracy_per_class(class) = (tp + tn) / (tp + tn + fp + fn) * 100;
            end
        end

        
        
        function stop = check_early_stop(obj, valid_loss)
            stop = false;
            if valid_loss < obj.best_valid_loss
                obj.best_valid_loss = valid_loss;
                obj.stop_counter = 0;
            else
                obj.stop_counter = obj.stop_counter + 1;
                if obj.stop_counter >= obj.early_stop_limit
                    stop = true;
                end
            end
        end
    end
end