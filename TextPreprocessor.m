classdef TextPreprocessor < handle
    properties
        vector_map
        tag_map
        output_dim = 4
    end
    
    methods
        function obj = TextPreprocessor()
            obj.init_tag_mapping();
            obj.vector_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
        
        function init_tag_mapping(obj)
            obj.tag_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
            
            noun_codes = [21, 22, 23, 24, 25, 31, 32];
            for code = noun_codes
                obj.tag_map(code) = 1;
            end
            
            verb_codes = [37, 38, 39, 40, 41, 42];
            for code = verb_codes
                obj.tag_map(code) = 2;
            end
            
            mod_codes = [12, 13, 14, 33, 34, 35];
            for code = mod_codes
                obj.tag_map(code) = 3;
            end
        end

        function load_vectors(obj, filepath)
            fprintf('Loading word vectors from %s...\n', filepath);
        
            opts = detectImportOptions(filepath);
            opts.SelectedVariableNames = {'word', 'embedding'};
            data = readtable(filepath, opts);
        
            for i = 1:height(data)
                word = char(data.word(i));
                vector_str = char(data.embedding(i));
                vector_str = regexprep(vector_str, '[\[\]]', '');
                vector_values = str2double(strsplit(strtrim(vector_str), ' '));
                obj.vector_map(word) = vector_values;
            end
        
            obj.vector_map('<start>') = zeros(1, 64);
            fprintf('Loaded %d word vectors\n', obj.vector_map.Count);
        end
        
        function [inputs, outputs] = load_data(obj, filepath)
            fprintf('Loading data from %s...\n', filepath);
        
            opts = detectImportOptions(filepath);
            opts.SelectedVariableNames = {'tokens', 'pos_tags'};
            data = readtable(filepath, opts);
        
            inputs = cell(height(data), 1);
            outputs = cell(height(data), 1);
        
            for i = 1:height(data)
                tokens_str = char(data.tokens(i));
                tokens_str = regexprep(tokens_str, '^\[|\]$', '');
                tokens_str = strrep(tokens_str, '''', '');
                tokens = strsplit(tokens_str, ', ');
        
                tags_str = char(data.pos_tags(i));
                tags_str = regexprep(tags_str, '^\[|\]$', '');
                tags = str2double(strsplit(tags_str, ', '));
        
                consolidated_tags = zeros(size(tags));
                for j = 1:length(tags)
                    if obj.tag_map.isKey(tags(j))
                        consolidated_tags(j) = obj.tag_map(tags(j));
                    else
                        consolidated_tags(j) = 4;
                    end
                end
        
                inputs{i} = tokens;
                outputs{i} = consolidated_tags;
            end
        
            fprintf('Loaded %d sequences\n', length(inputs));
        end

        function [batch_inputs, batch_outputs] = prepare_mini_batch(obj, inputs, outputs, batch_size)
            num_sequences = length(inputs);
            indices = randperm(num_sequences, min(batch_size, num_sequences));
            batch_inputs = inputs(indices);
            batch_outputs = outputs(indices);
        end
        
        function vector = get_vector(obj, word)
            if obj.vector_map.isKey(word)
                vector = obj.vector_map(word);
            else
                vector = zeros(1, 64);
            end
        end
    end
end