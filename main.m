function main()    
    training_path = 'train_data.csv';
    validation_path = 'valid_data.csv';
    testing_path = 'test_data.csv';
    vector_path = 'embeddings.csv';
    
    preprocessor = TextPreprocessor();
    preprocessor.load_vectors(vector_path);
    tagger = StateSpaceTagger(preprocessor.vector_map);
    optimizer = StateSpaceOptimizer(tagger, preprocessor);
    
    [train_inputs, train_outputs] = preprocessor.load_data(training_path);
    [valid_inputs, valid_outputs] = preprocessor.load_data(validation_path);
    [test_inputs, test_outputs] = preprocessor.load_data(testing_path);
    
    optimizer.optimize(train_inputs, train_outputs, valid_inputs, valid_outputs);
    
    [test_loss, test_preds, test_true] = optimizer.evaluate(test_inputs, test_outputs);
    metrics = optimizer.compute_metrics(test_preds, test_true);
    
    fprintf('\nFinal Model Performance:\n');
    fprintf('Overall Precision: %.2f%%\n', mean(metrics.precision));
    fprintf('Overall Recall: %.2f%%\n', mean(metrics.recall));
    fprintf('Overall Accuracy: %.2f%%\n', metrics.accuracy);
        
end