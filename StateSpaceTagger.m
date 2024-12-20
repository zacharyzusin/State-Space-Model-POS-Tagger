classdef StateSpaceTagger < handle
    properties
        hidden_dim = 64
        output_dim = 4
        context_size = 4
        
        state_matrix
        input_proj
        output_proj
        residual_conn
        time_step
        
        disc_state
        disc_input
        disc_output
        disc_residual
        kernel
        
        class_weights
        class_bias
        
        vector_map
        unknown_vector_map
    end
    
    methods
        function obj = StateSpaceTagger(vector_map)
            obj.vector_map = vector_map;
            obj.unknown_vector_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
            
            obj.init_parameters();
            
            obj.class_weights = randn(obj.output_dim, obj.hidden_dim) * sqrt(2/obj.hidden_dim);
            obj.class_bias = zeros(obj.output_dim, 1);
            
            obj.compute_kernel();
        end
        
        function init_parameters(obj)
            N = obj.hidden_dim;
            
            obj.state_matrix = zeros(N);
            for n = 1:N
                for k = 1:N
                    if n > k
                        obj.state_matrix(n,k) = -sqrt((2*n+1)/(2*k+1));
                    elseif n == k
                        obj.state_matrix(n,k) = -(n+1);
                    end
                end
            end
            
            obj.input_proj = randn(N, 1) * 0.1;
            obj.output_proj = randn(N, 1) * 0.1;
            obj.residual_conn = randn(1, 1) * 0.1;
            obj.time_step = 1.0;
            
            I = eye(N);
            obj.disc_state = (I - obj.time_step/2 * obj.state_matrix)^(-1) * ...
                           (I + obj.time_step/2 * obj.state_matrix);
            obj.disc_input = (I - obj.time_step/2 * obj.state_matrix)^(-1) * ...
                          obj.time_step * obj.input_proj;
            obj.disc_output = obj.output_proj;
            obj.disc_residual = obj.residual_conn;
        end
        
        function compute_kernel(obj)
            obj.kernel = zeros(1, obj.context_size);
            curr_term = obj.disc_output' * obj.disc_input;
            obj.kernel(1) = curr_term;
            
            curr_power = obj.disc_input;
            for i = 2:obj.context_size
                curr_power = obj.disc_state * curr_power;
                curr_term = obj.disc_output' * curr_power;
                obj.kernel(i) = curr_term;
            end
        end
        
        function [context_windows, num_windows] = process_sequence(obj, tokens)
            padding_size = obj.context_size - 1;
            padded_tokens = cell(1, padding_size + length(tokens));
            padded_tokens(1:padding_size) = {'<start>'};
            padded_tokens(padding_size+1:end) = tokens;
            
            num_windows = length(tokens);
            context_windows = zeros(num_windows, obj.context_size, obj.hidden_dim);
            
            for i = 1:num_windows
                window_tokens = padded_tokens(i:i+obj.context_size-1);
                
                for j = 1:obj.context_size
                    token = lower(window_tokens{j});
                    
                    if strcmp(token, '<start>')
                        context_windows(i,j,:) = zeros(1, obj.hidden_dim);
                    elseif obj.vector_map.isKey(token)
                        context_windows(i,j,:) = obj.vector_map(token);
                    else
                        if ~obj.unknown_vector_map.isKey(token)
                            random_vector = randn(1, obj.hidden_dim) * sqrt(2/obj.hidden_dim);
                            obj.unknown_vector_map(token) = random_vector;
                        end
                        context_windows(i,j,:) = obj.unknown_vector_map(token);
                    end
                end
            end
        end
        
        function [predictions, hidden_states] = forward(obj, x)
            [batch_size, ~, ~] = size(x);
            hidden_states = zeros(batch_size, obj.hidden_dim);
            
            for b = 1:batch_size
                seq = squeeze(x(b,:,:));
                h = zeros(obj.hidden_dim, 1);
                
                for i = 1:obj.context_size
                    u = seq(i,:)';
                    for j = 1:i
                        h = h + obj.kernel(j) * u;
                    end
                    h = h + obj.disc_residual * u;
                end
                hidden_states(b,:) = h';
            end
            
            logits = hidden_states * obj.class_weights' + repmat(obj.class_bias', batch_size, 1);
            predictions = softmax(logits')';
        end 
    end
end