import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from shared.component_logger import component_logger as logger
from gdn import GDN
from scipy.stats import iqr
from metrics.eval_metrics import precision, recall, f1_score

class GraphBasedAnomalyDetection(nn.Module):
    def __init__(self, params):
        super(GraphBasedAnomalyDetection, self).__init__()
        self.params = params
        self.seed_for_experiment(params.seed)
        # 1. Create a fully connected graph for GDN. 
        self.graph_deviation_network = GDN(self.params.edge_index, self.params.node_num, self.params.dim, self.params.slide_win, 
                                        self.params.out_layer_num, self.params.out_layer_inter_dim, self.params.topk).to(self.params.device)

        # 2. Create the optimizers, Adam optimizer with learning rate of 0.001.
        self.optimizer = torch.optim.Adam(self.graph_deviation_network.parameters(), lr=0.001, weight_decay= params.decay)
        logger.log("Trainable Parameters in Generator: {:,}".format(self.parameters_count(self.graph_deviation_network)))


    def seed_for_experiment(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def parameters_count(self, model):
        """
        params:
            model: A neural network architecture to count trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, step, path):
        """
        params:
            step: step of the model
            path: path to save the model
        """
        try:
            torch.save({
                'step': step,
                'model_state_dict': self.graph_deviation_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

        except Exception as e:
            raise ValueError("Invalid model or path name, with error: {}".format(e))

    def load_model(self, path):
        """
        Load model from the path.

        params:
            path: path to the model
        """
        try:
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.gen_optim.load_state_dict(checkpoint['optimizer_state_dict'])

        except Exception as e:
            raise ValueError("Invalid model or path name, with error: {}".format(e))



    def train(self, train_dataloader, valid_dataloader):
            """
            Train the model on the training set and evaluate on the validation set.
            """
            params = self.params
            train_loss_list = []
            
            stop_improve_count = 0
            min_loss = 1e+8
            # Training the model with MSE loss function.

            for i in range(params.epoch):
                current_loss = 0
                self.graph_deviation_network.train() #setting the model to train mode
                for step, (x, labels, attack_labels, edge_index) in enumerate(train_dataloader):
                    x, labels, edge_index = [item.float().to(params.device) for item in [x, labels, edge_index]]
                    self.optimizer.zero_grad()
                    out = self.graph_deviation_network(x)
                    out = out.float().to(params.device)
                    loss = F.mse_loss(out, labels, reduction='mean')        
                    loss.backward()
                    self.optimizer.step()
                    train_loss_list.append(loss.item())
                    current_loss += loss.item()
                
                val_loss, val_result = self.test(valid_dataloader)
                
                self.save_model(step, os.path.join(params.model_save_path, "graph_deviation_network.pth"))
                logger.log("Epoch: {}; mse_train: {}; mse_valid: {}".format(i + 1, np.round(current_loss/len(train_dataloader), 4), np.round(val_loss, 4)))

                if val_loss < min_loss:
                    min_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1
                
                if stop_improve_count > params.early_stop_win:
                    logger.log("Early stopping at Epoch {}".format(i))
                    break

            
            logger.log("Tuning the model for normalized error scores")
            self.normalized_error_scores = self.tune(valid_dataloader)
            logger.log("Finished training")
            return train_loss_list
    
    def test(self, dataloader):
        """
        Given the dataloaders which consists the grouth truth information (attack_labels) in the form of (x, targets, attack_labels, edge_index) from dataloaders,
        test the model and return the mse loss and the prediction of the model for performance assessments.

        Parameters
        ----------
            dataloader: PyTorch dataloader with test data, can work with train/valid. Data is in the form of (x, targets, attack_labels, edge_index)
        """
        params = self.params
        device = params.device
        test_loss_list = []
        test_predicted_list = []
        test_ground_list = []
        test_labels_list = []
        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []

        self.graph_deviation_network.eval()

        i = 0
        acu_loss = 0
        for x, y, labels, edge_index in dataloader:
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            
            with torch.no_grad():
                predicted = self.graph_deviation_network(x)
                predicted = predicted.float().to(device)
                loss = F.mse_loss(predicted, y, reduction='mean')
                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
            
            
            test_loss_list.append(loss.item())
            acu_loss += loss.item()
            i += 1

        test_predicted_list = t_test_predicted_list.tolist()        
        test_ground_list = t_test_ground_list.tolist()        
        test_labels_list = t_test_labels_list.tolist()      
        
        avg_loss = sum(test_loss_list)/len(test_loss_list)
        return avg_loss, np.array([test_predicted_list, test_ground_list, test_labels_list])

        
    
    def tune(self, val_dataloader):
        """ 
        Tune the model using the validation dataloader. Find the threshold value by computing the normalized error scores.

        Parameters
        ----------
            val_dataloader: PyTorch dataloader with validation data. Data is in the form of (x, targets, attack_labels, edge_index)
        """
        _, val_result = self.test(val_dataloader)
        normalized_error_scores = self.get_full_err_scores(val_result)
        return normalized_error_scores


    def predict(self, test_dataloader):
        """    
        Predict the labels of the test data using the model.

        Parameters
        ----------
            test_dataloader: PyTorch dataloader with test data. Data is in the form of (x, targets, attack_labels, edge_index)
        """
        params = self.params
        _, test_result = self.test(test_dataloader)
        normal_scores = self.normalized_error_scores
        topk = 1
        total_topk_err_scores = []
        total_err_scores = self.get_full_err_scores(test_result)
        total_features = total_err_scores.shape[0]
        topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

        threshold = self.get_anomalous_threshold(normal_scores)

        pred_labels = np.zeros(len(total_topk_err_scores))
        pred_labels[total_topk_err_scores > threshold] = 1
        indices = np.argwhere(pred_labels == 1).flatten() + params.slide_win
        logger.log("Amount of outliers detetced: {}".format(len(indices)))
        return indices

    def get_err_median_and_iqr(self, predicted, groundtruth):
        """
        Given the predicted and groundtruth labels, compute the median and IQR of the error scores.
        """
        np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

        err_median = np.median(np_arr)
        err_iqr = iqr(np_arr)

        return err_median, err_iqr

    def get_err_scores(self, test_res):
        """
        Given the test result, compute the error scores.
        """
        test_predict, test_gt = test_res
        n_err_mid, n_err_iqr = self.get_err_median_and_iqr(test_predict, test_gt)

        test_delta = np.abs(np.subtract(
                            np.array(test_predict).astype(np.float64), 
                            np.array(test_gt).astype(np.float64)
                        ))
        epsilon=1e-2

        err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3
        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

        return smoothed_err_scores

    def get_full_err_scores(self, test_result):
        """
        Given the test result, compute the error scores.
        """
        np_test_result = np.array(test_result)
        all_scores =  None
        feature_num = np_test_result.shape[-1]

        for i in range(feature_num):
            test_re_list = np_test_result[:2,:,i]
            normalized_error = self.get_err_scores(test_re_list)

            if all_scores is None:
                all_scores = normalized_error
            else:
                all_scores = np.vstack((
                    all_scores,
                    normalized_error
                ))
        return all_scores

    def get_anomalous_threshold(self, normal_scores):
        """
        Compute the threshold value for the anomaly detection.
        """
        #print(normal_scores.shape)
        threshold= np.max(normal_scores)
        #print(threshold)
        return threshold

    def get_val_performance_data(self, total_err_scores, normal_scores, gt_labels, topk=1):
        """
        Given the error scores, compute the performance metrics.
        """
        total_topk_err_scores = []
        total_features = total_err_scores.shape[0]
        topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

        threshold = self.get_anomalous_threshold(normal_scores)

        pred_labels = np.zeros(len(total_topk_err_scores))
        pred_labels[total_topk_err_scores > threshold] = 1

        for i in range(len(pred_labels)):
            pred_labels[i] = int(pred_labels[i])
            gt_labels[i] = int(gt_labels[i])
        
        pre = precision(gt_labels, pred_labels)
        rec = recall(gt_labels, pred_labels)
        f1 = f1_score(gt_labels, pred_labels)

        return f1, pre, rec, threshold

    def eval_model(self, test_dataloader, valid_dataloader):
        """
        Only works on labelled data. Given the test and validation dataloaders, evaluate the model.
        """
        _, test_result = self.test(test_dataloader)
        _, val_result = self.test(valid_dataloader)

        normalized_error_scores = self.get_full_err_scores(val_result)
        
        test_labels = test_result[2, :, 0].tolist()
        test_scores = self.get_full_err_scores(test_result) #action indexes -> test_result[:2, :, _]
        top1_val_info = self.get_val_performance_data(test_scores, normalized_error_scores, test_labels, topk=1)

        logger.log("------------Evaluation Report------------")
        logger.log(f'Precision: {top1_val_info[1]}')
        logger.log(f'Recall: {top1_val_info[2]}')
        logger.log(f'F1 score: {top1_val_info[0]}')
        pass