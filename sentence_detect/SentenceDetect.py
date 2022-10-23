from transformers import XLNetTokenizer, XLNetModel
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import torch
import random
import math
from torch.autograd.grad_mode import no_grad

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

class SentenceDetectionModel(nn.Module):
    def __init__(self, num_sentences, train_data_file, model_state_dict):
        super(SentenceDetectionModel, self).__init__()
        self.num_sentences = num_sentences
        self.embedding_size = 768
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)
        self.xlnetModel = XLNetModel.from_pretrained('xlnet-base-cased').to(self.device, dtype=torch.float, non_blocking=True)
        self.multi_head_attention_inter_sentence = nn.MultiheadAttention(self.embedding_size, 16)

        self.hidden = nn.Linear((self.num_sentences) * self.embedding_size, 4000, bias=True).to(self.device, dtype=torch.float, non_blocking=True)
        self.activation = nn.Tanh().to(self.device, dtype=torch.float, non_blocking=True)
        self.dropout = nn.Dropout(0.5).to(self.device, dtype=torch.float, non_blocking=True)
        
        self.hidden1 = nn.Linear(4000, 2000, bias=True).to(self.device, dtype=torch.float, non_blocking=True)
        self.activation1 = nn.Tanh().to(self.device, dtype=torch.float, non_blocking=True);
        self.dropout1 = nn.Dropout(0.1).to(self.device, dtype=torch.float, non_blocking=True);

        self.hidden2 = nn.Linear(2000, 1000, bias=True).to(self.device, dtype=torch.float, non_blocking=True)
        self.activation2 = nn.Tanh().to(self.device, dtype=torch.float, non_blocking=True);
        self.dropout2 = nn.Dropout(0.1).to(self.device, dtype=torch.float, non_blocking=True);

        self.hidden3 = nn.Linear(1000, 100, bias=True).to(self.device, dtype=torch.float, non_blocking=True)
        self.activation3 = nn.Tanh().to(self.device, dtype=torch.float, non_blocking=True);
        self.dropout3 = nn.Dropout(0.1).to(self.device, dtype=torch.float, non_blocking=True);

        self.out = nn.Linear(100, self.num_sentences, bias=True).to(self.device, dtype=torch.float, non_blocking=True)
        self.softmax = nn.Softmax(-1).to(self.device, dtype=torch.float, non_blocking=True)

        #Freeze the parameters of the model.
        #for param in self.xlnetModel.base_model.parameters():
            #param.requires_grad = False
        
        
        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.validation_error = None
        self.train_error =  None
        
        if(train_data_file is not None):
            data = self.extract_data_from_file(train_data_file)
            random.shuffle(data)
            train_size = math.floor(0.8*len(data))
            val_size = math.floor(0.1*len(data))
            train, val ,test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]
            self.train_dataset = train
            self.test_dataset = test
            self.validation_dataset = val
            self.validation_error = []
            self.train_error = []
            
        if(model_state_dict is not None):
            self.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cpu')))
            
        if(model_state_dict is None and train_data_file is None):
            raise("Incorrect Parameters. Atlease one of model_state_dict or train_data_file required")
        
    def forward(self, input):
        input = input.to(self.device)
        
        model_outputs = self.xlnetModel(**input)

        sentence_section_output = model_outputs.last_hidden_state[:, 0:480, :]
        sentence_embeddings = torch.zeros((1, self.num_sentences, self.embedding_size))
        
        for i in range(self.num_sentences):
            sentence_start = (i*48)
            sentence_end = ((i+1)*48)
            sentence = sentence_section_output[:,  sentence_start:sentence_end  , :];
            sentence = sentence.to(self.device).float()
            sentence_tensor = torch.mean(sentence, 1)
            sentence_embeddings[0][i] = sentence_tensor
      
        sentence_embeddings = sentence_embeddings.to(self.device).float()
        concat_sentence_attention = sentence_embeddings + self.multi_head_attention_inter_sentence(sentence_embeddings, sentence_embeddings,sentence_embeddings)[0]

        
        hidden = self.dropout(self.activation(self.hidden(
            concat_sentence_attention.reshape(1, (self.num_sentences) * self.embedding_size))))
        hidden1 = self.dropout1(self.activation1(self.hidden1(hidden)))
        hidden2 = self.dropout2(self.activation2(self.hidden2(hidden1)))
        hidden3 = self.dropout3(self.activation3(self.hidden3(hidden2)))
                
        output = self.out(hidden3)
        classification = self.softmax(output)
        return classification
        
    def train_model(self, epochs):
        
        #some code borrowed from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        train_data = self.train_dataset
        loss_function = nn.BCELoss(reduction='sum')
        optimiser = Adam(self.parameters(), lr=2e-5)
        optimizer_to(optimiser, self.device)
        
        for i in range(epochs):
            running_loss = 0
            print("starting epoch " + str(i))
            random.shuffle(train_data)
            
            self.train()
            net_train_loss = 0;
            for index, ((question, sentences), (data, label)) in enumerate(train_data):
                optimiser.zero_grad()
                
                out = self(data)
                label = label.to(self.device)
                loss = loss_function(out, label)
                loss.backward()
                optimiser.step()
                
                running_loss += loss.item()
                net_train_loss += loss.item()
                if index % 10 == 9:    # print every 10 mini-batches
                    print(f'[{index + 1}, {index + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
            
            self.train_error.append(net_train_loss)
            self.eval()
            with no_grad():
              net_loss = 0;
              for index, ((question, sentences), (data, label)) in enumerate(self.validation_dataset):
                output = self(data)
                label = label.to(self.device)
                loss = loss_function(output, label)
                net_loss = net_loss + loss.item()
              self.validation_error.append(net_loss)
              print("Validation : " + str(net_loss))
    
    def tokenize_sentences(self, sentences, question):
        tokenized_question = self.tokenizer(question, max_length = 32, padding = 'max_length', truncation = True, add_special_tokens=False, return_tensors='pt')
        tokenized_sentences = self.tokenizer(sentences, max_length = 48, padding = 'max_length', truncation = True, add_special_tokens=False, return_tensors='pt')
        tokenized_sentences['input_ids'] = tokenized_sentences.input_ids.reshape(1, tokenized_sentences.input_ids.shape[0] * tokenized_sentences.input_ids.shape[1])
        tokenized_sentences['token_type_ids'] = tokenized_sentences.token_type_ids.reshape(1, tokenized_sentences.token_type_ids.shape[0] * tokenized_sentences.token_type_ids.shape[1])
        tokenized_sentences['attention_mask'] = tokenized_sentences.attention_mask.reshape(1, tokenized_sentences.attention_mask.shape[0] * tokenized_sentences.attention_mask.shape[1])
        tokenized_sentences['input_ids'] = torch.cat((tokenized_sentences['input_ids'], self.tokenizer("", return_tensors='pt').input_ids[:,0].reshape(1,1)), dim = 1)
        tokenized_sentences['token_type_ids'] = torch.cat((tokenized_sentences['token_type_ids'], torch.tensor([[0]])), dim = 1)
        tokenized_sentences['attention_mask'] = torch.cat((tokenized_sentences['attention_mask'], torch.tensor([[1]])), dim = 1)
        tokenized_sentences['input_ids'] = torch.cat((tokenized_sentences['input_ids'], tokenized_question.input_ids), dim=1)
        tokenized_sentences['attention_mask'] = torch.cat((tokenized_sentences['attention_mask'], tokenized_question.attention_mask), dim=1)
        tokenized_question.token_type_ids[tokenized_question.token_type_ids == 0] = 1
        tokenized_sentences['token_type_ids'] = torch.cat((tokenized_sentences['token_type_ids'], tokenized_question.token_type_ids), dim=1)
        tokenized_sentences['input_ids'] = torch.cat((tokenized_sentences['input_ids'], self.tokenizer("", return_tensors='pt').input_ids), dim = 1)
        tokenized_sentences['token_type_ids'] = torch.cat((tokenized_sentences['token_type_ids'], torch.tensor([[1,2]])), dim = 1)
        tokenized_sentences['attention_mask'] = torch.cat((tokenized_sentences['attention_mask'], self.tokenizer("", return_tensors='pt').attention_mask), dim = 1)
        return tokenized_sentences
          
    def extract_data_from_file(self, file_path):
        data = pd.read_excel(file_path)
        train_data = []
        for row in data.iloc:
            labels = row["Valid Sentences"]
            label_tensor = torch.zeros(10)
            for label in str(labels).split(","):
                parsed_label = int(label.strip())
                label_tensor[parsed_label] = 1
            label_tensor = label_tensor.reshape(1, 10) 

            question = row["question"]
            sentences = self.split_passages(str(row["Passage (10 sentences)"]))
                
            if(len(sentences) != 10):
                print(sentences)
                raise("Invalid number of sentences")
            
            datapoint = ((question, sentences), (self.tokenize_sentences(sentences, question), label_tensor))
            train_data.append(datapoint)
        return train_data
    
    def split_passages(self, passage):
        sentences = []
        for sent in passage.split('\n'):
            sentences.append(sent)
        return sentences;
    
    def get_predicted_sentences(self, passage, question):
        self.eval()
        sentences = self.split_passages(passage)
        tokenized_sentences = self.tokenize_sentences(sentences, question)
        output = self(tokenized_sentences)
        result_sentence_indexes = ((output[0] > 0.1).nonzero().reshape(1, (output[0] > 0.1).nonzero().shape[0]))
        
        result = []
        print(result_sentence_indexes)
        print(sentences)
        for result_sentence_index in result_sentence_indexes[0].tolist():
            result.append(sentences[result_sentence_index])
        return result
            




            
            
                
                
        
        
        
    
