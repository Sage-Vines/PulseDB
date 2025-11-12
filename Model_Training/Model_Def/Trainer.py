# from datetime import datetime
# import time
# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import progressbar as PB
# from sklearn.metrics import r2_score
# from torch.utils.tensorboard import SummaryWriter as SW
# from io import StringIO
# import sys

# # R-Squared
# def R2(y_true, y_pred):
#     return r2_score(y_true, y_pred)

# # Mean error
# def ME(y_true, y_pred):
#     return np.mean(y_true-y_pred)

# # Standard deviation of error
# def SD(y_true, y_pred):
#     return np.std(y_true-y_pred)


# widgets = [
#     PB.Bar(),
#     PB.Counter(),
#     ' ',
#     PB.Percentage(),
#     ' ',
#     PB.DynamicMessage('Batch_BP_Loss'),
#     ' ',
#     PB.ETA()
# ]


# class Model_Trainer:
#     def __init__(self, model, criterion_BP, optimizer_BP, device, settings_yml, batch_size=32, num_epochs=100, save_states=False, save_final=False):

#         self.Model_Running = model.to(device)
#         self.Model_BestTest = []
#         self.BP_Loss_Fun = criterion_BP
#         self.Optimizer_BP = optimizer_BP
#         self.Num_Epoch = num_epochs
#         self.Train_Batchsize = batch_size
#         self.Device = device
#         self.Save_States = save_states
#         self.Save_Final = save_final
#         self.YMLSettings = settings_yml

#     def Model_Info(self):
#         model = self.Model_Running
#         print('-' * 10)
#         print('Model Structure:')
#         print(model)
#         num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print('Trainable parameters: {}'.format(num))
#         print('Settings')
#         for item, setting in self.YMLSettings.items():
#             print(item, ':', setting)
#         print('-' * 10)

#     def Set_Dataset(self, train_set, test_set=[]):
#         self.Train_Set = train_set
#         self.Test_Set_List = test_set

#     def Train_Model(self):
#         TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
#         ModelID = TimeID[-6:]
        
#         Start_Epoch = 1
#         batchcounter = 1
#         batchrecordcounter = 1

  
#         # Print model info to command line
#         Writer = SW(os.path.join('TensorBoard', TimeID))
#         print('ModelID: '+ModelID)
#         self.Model_Info()
#         # Print model info to string so it can be documented in TensorBoard
#         ##################################################################################
#         save_stdout = sys.stdout
#         result = StringIO()
#         sys.stdout = result
#         print('ModelID: '+ModelID)
#         self.Model_Info()
#         sys.stdout = save_stdout
#         Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))
#         ##################################################################################

#         # Set up data loaders for the training and testing sets
#         Train = data.DataLoader(
#             self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
#         Test_Names = []
#         Test_List = []
#         for name, testdata in self.Test_Set_List.items():
#             Test_Names.append(name)
#             Test_List.append(data.DataLoader(testdata, batch_size=128))

#         Start_Time = time.time()
#         # Set up keybaord interrupt, so when training process is interrupted, the model can still be save to files
#         Interrupt = False

#         Train_Batch = self.Train_Batch

#         for Epoch in range(Start_Epoch, Start_Epoch+self.Num_Epoch):
#             try:
#                 print('Epoch {}/{}'.format(Epoch, Start_Epoch+self.Num_Epoch-1))
#                 print('-' * 10)

#                 # Training Phase
#                 Epoch_BP_Train_Loss = []
#                 k = 0
#                 Epoch_BP_Preds = []
#                 Epoch_BP_Labels = []
#                 with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
#                     for inputs, BP_labels in Train:
#                         # Accumulate batch-by-batch training outputs
#                         BP_loss, BP_Outputs = Train_Batch(inputs, BP_labels)
#                         Epoch_BP_Labels.append(
#                             BP_labels.cpu().detach().numpy())
#                         Epoch_BP_Preds.append(
#                             BP_Outputs.cpu().detach().numpy())
#                         # Update on the progress bar
#                         bar.update(k, Batch_BP_Loss=BP_loss)
#                         k += 1
#                         batchcounter += 1
#                         # Save training loss of each batch to TensorBoard
#                         if not batchcounter % 100:
#                             Writer.add_scalar(
#                                 'Batch_BP_Loss', BP_loss, batchrecordcounter)
#                             batchrecordcounter += 1  # each count is 100 batches
                    
#                     # At the end of each epoch, calculate the error metrics on the training set
#                     Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
#                     Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
#                     Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
#                     Epoch_Train_ME = ME(Epoch_BP_Labels, Epoch_BP_Preds)
#                     Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)

#                     # Calculate the training loss
#                     Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(
#                         Epoch_BP_Labels), torch.from_numpy(Epoch_BP_Preds))
                    
#                     # Print a summary of training error metrics
#                     print('Epoch BP Training Loss: {:e} R2: {}'.format(
#                         Epoch_BP_Train_Loss, Epoch_Train_R2))
                    
#                     # Save a checkpoint
#                     if self.Save_States:
#                         self.Save_Checkpoint(
#                             ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)

#                 # Write training results to TensorBoard
#                 Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
#                 Writer_R2_Dict = {'Train': Epoch_Train_R2}
#                 Writer_ME_Dict = {'Train': Epoch_Train_ME}
#                 Writer_SD_Dict = {'Train': Epoch_Train_SD}

#                 # Testing Phase
                
#                 # Rrun testing with the current model on each of the testing sets
#                 for name, Test in zip(Test_Names, Test_List):
#                     Test_Name = name
#                     Epoch_Test_Loss = []
#                     Epoch_Preds = []
#                     Epoch_Labels = []
#                     # Accumulate predictions batch by batch
#                     for inputs, labels in Test:
#                         Loss_Per_Batch, Outputs = self.Test_Batch(
#                             inputs, labels)
#                         Epoch_Test_Loss.append(Loss_Per_Batch)
#                         Epoch_Labels.append(labels.cpu().detach().numpy())
#                         Epoch_Preds.append(Outputs.cpu().detach().numpy())
#                     # Calculate the error metrics
#                     Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
#                     Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

#                     Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(
#                         Epoch_Labels), torch.from_numpy(Epoch_Preds))
#                     Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
#                     Epoch_Test_ME = ME(Epoch_Labels, Epoch_Preds)
#                     Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)
#                     # Print a summary
#                     print(
#                         'Epoch '+Test_Name + ' Loss: {:e} R2: {}'.format(Epoch_Test_Loss, Epoch_Test_R2))
#                     # Write to TensorBoard
#                     Writer_Loss_Dict.update({Test_Name: Epoch_Test_Loss})
#                     Writer_R2_Dict.update({Test_Name: Epoch_Test_R2})
#                     Writer_ME_Dict.update({Test_Name: Epoch_Test_ME})
#                     Writer_SD_Dict.update({Test_Name: Epoch_Test_SD})

#                 ################################################################################################
#                 Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
#                 Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
#                 Writer.add_scalars('ME', Writer_ME_Dict, Epoch)
#                 Writer.add_scalars('SD', Writer_SD_Dict, Epoch)
#                 ################################################################################################

#             # If the training is manually stopped by keyboard interruption
#             except KeyboardInterrupt:
#                 print('Earlystopped by interrupt at epoch {:d}'.format(Epoch))
#                 Interrupt = True
#                 break
#         Writer.close()
#         time_elapsed = time.time() - Start_Time
#         print('Training complete in {:.0f}m {:.0f}s'.format(
#             time_elapsed // 60, time_elapsed % 60))
#         # Save the model
#         self.Save_Checkpoint(ModelID, TimeID, Epoch,
#                              batchcounter, batchrecordcounter, savemodel=True)
#         if Interrupt:
#             raise KeyboardInterrupt

#     # Forward propagation of a batch in training mode
#     def Train_Batch(self, inputs, BP_labels):
#         self.Model_Running.train()
#         inputs = inputs.float().to(self.Device)
#         BP_labels = BP_labels.float().to(self.Device)

#         self.Model_Running.zero_grad()
#         BP_outputs = self.Model_Running(inputs)
#         BP_loss = self.BP_Loss_Fun(BP_outputs, BP_labels)
#         BP_loss_report = BP_loss.item()
#         BP_loss.backward()
#         self.Optimizer_BP.step()

#         return BP_loss_report, BP_outputs
#     # Forward propagation of a batch in testing mode (inference)
#     def Test_Batch(self, inputs, labels):

#         self.Model_Running.eval()
#         inputs = inputs.float().to(self.Device)
#         labels = labels.float().to(self.Device)
#         with torch.no_grad():
#             # When testing, display only the reconstruction loss
#             BP_outputs = self.Model_Running(inputs)
#             loss = self.BP_Loss_Fun(BP_outputs, labels)
#         return loss.item(), BP_outputs
#     # Save a checkpoint
#     def Save_Checkpoint(self, modelID, timeID, epoch, batchcounter, batchrecordcounter, savemodel=False):
#         # Save a dict for every epoch
#         foldername = modelID
#         if not os.path.isdir(foldername):
#             os.mkdir(foldername)
#         torch.save({'model_id': modelID,
#                     'time_id': timeID,
#                     'model_state_dict': self.Model_Running.state_dict(),
#                     'optimizer_state_dict': self.Optimizer_BP.state_dict(),
#                     'epoch': epoch,
#                     'batchcounter': batchcounter,
#                     'batchrecordcounter': batchrecordcounter,
#                     }, os.path.join(foldername, 'checkpoint_epoch_{}.pth'.format(epoch)))
#         if savemodel:
#             torch.save(self.Model_Running, os.path.join(
#                 foldername, 'trained_model.pth'))



from datetime import datetime
import time
import torch
import torch.utils.data as data
import os
import numpy as np
import progressbar as PB
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter as SW
from io import StringIO
import sys


# Utility metric functions
def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def ME(y_true, y_pred):
    return np.mean(y_true - y_pred)

def SD(y_true, y_pred):
    return np.std(y_true - y_pred)


# Progress bar configuration
widgets = [
    PB.Bar(),
    PB.Counter(),
    ' ',
    PB.Percentage(),
    ' ',
    PB.DynamicMessage('Batch_BP_Loss'),
    ' ',
    PB.ETA()
]


# Model Trainer Class
class Model_Trainer:
    def __init__(self, model, criterion_BP, optimizer_BP, device, settings_yml,
                 batch_size=32, num_epochs=100, save_states=False, save_final=False):

        self.Model_Running = model.to(device)
        self.Model_BestTest = []
        self.BP_Loss_Fun = criterion_BP
        self.Optimizer_BP = optimizer_BP
        self.Num_Epoch = num_epochs
        self.Train_Batchsize = batch_size
        self.Device = device
        self.Save_States = save_states
        self.Save_Final = save_final
        self.YMLSettings = settings_yml


    # --------------------------------------------------------
    def Model_Info(self):
        model = self.Model_Running
        print('-' * 10)
        print('Model Structure:')
        print(model)
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(num))
        print('Settings')
        for item, setting in self.YMLSettings.items():
            print(item, ':', setting)
        print('-' * 10)


    # --------------------------------------------------------
    def Set_Dataset(self, train_set, test_set=[]):
        self.Train_Set = train_set
        self.Test_Set_List = test_set


    # Main training loop
    def Train_Model(self):
        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
        ModelID = TimeID[-6:]
        Start_Epoch = 1
        batchcounter = 1
        batchrecordcounter = 1

        # TensorBoard writer
        Writer = SW(os.path.join('TensorBoard', TimeID))
        print('ModelID: ' + ModelID)
        self.Model_Info()

        # Capture model info for TensorBoard text log
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: ' + ModelID)
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))

        # Data loaders
        Train = data.DataLoader(self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
        Test_Names = []
        Test_List = []
        for name, testdata in self.Test_Set_List.items():
            Test_Names.append(name)
            Test_List.append(data.DataLoader(testdata, batch_size=128))

        Start_Time = time.time()
        Interrupt = False
        Train_Batch = self.Train_Batch
        
        # Epoch Loop
        for Epoch in range(Start_Epoch, Start_Epoch + self.Num_Epoch):
            try:
                print(f'Epoch {Epoch}/{Start_Epoch + self.Num_Epoch - 1}')
                print('-' * 10)

                # Training Phase
                k = 0
                Epoch_BP_Preds = []
                Epoch_BP_Labels = []
                with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
                    for inputs, BP_labels in Train:
                        BP_loss, BP_Outputs = Train_Batch(inputs, BP_labels)
                        Epoch_BP_Labels.append(BP_labels.cpu().detach().numpy())
                        Epoch_BP_Preds.append(BP_Outputs.cpu().detach().numpy())
                        bar.update(k, Batch_BP_Loss=BP_loss)
                        k += 1
                        batchcounter += 1
                        if not batchcounter % 100:
                            Writer.add_scalar('Batch_BP_Loss', BP_loss, batchrecordcounter)
                            batchrecordcounter += 1

                # Compute Training Metrics
                Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
                Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
                Epoch_BP_Preds = np.atleast_2d(Epoch_BP_Preds)
                Epoch_BP_Labels = np.atleast_2d(Epoch_BP_Labels)


                # Flexible handling for 1D or 2D output
                if Epoch_BP_Preds.ndim == 1 or Epoch_BP_Preds.shape[1] == 1:
                    Epoch_Train_R2 = {'SBP': R2(Epoch_BP_Labels, Epoch_BP_Preds)}
                    Epoch_Train_ME = {'SBP': ME(Epoch_BP_Labels, Epoch_BP_Preds)}
                    Epoch_Train_SD = {'SBP': SD(Epoch_BP_Labels, Epoch_BP_Preds)}
                else:
                    Epoch_Train_R2 = {
                        'SBP': R2(Epoch_BP_Labels[:, 0], Epoch_BP_Preds[:, 0]),
                        'DBP': R2(Epoch_BP_Labels[:, 1], Epoch_BP_Preds[:, 1])
                    }
                    Epoch_Train_ME = {
                        'SBP': ME(Epoch_BP_Labels[:, 0], Epoch_BP_Preds[:, 0]),
                        'DBP': ME(Epoch_BP_Labels[:, 1], Epoch_BP_Preds[:, 1])
                    }
                    Epoch_Train_SD = {
                        'SBP': SD(Epoch_BP_Labels[:, 0], Epoch_BP_Preds[:, 0]),
                        'DBP': SD(Epoch_BP_Labels[:, 1], Epoch_BP_Preds[:, 1])
                    }

                # Epoch_BP_Train_Loss = self.BP_Loss_Fun(
                #     torch.from_numpy(Epoch_BP_Preds),
                #     torch.from_numpy(Epoch_BP_Labels)
                # )
                
                Epoch_BP_Train_Loss = self.BP_Loss_Fun(
                    torch.from_numpy(Epoch_BP_Preds).float().to(self.Device),
                    torch.from_numpy(Epoch_BP_Labels).float().to(self.Device)
)               


                print(f'Epoch BP Training Loss: {Epoch_BP_Train_Loss:.6e}')
                for k, v in Epoch_Train_R2.items():
                    print(f'  {k}: R2={v:.4f} ME={Epoch_Train_ME[k]:.2f} SD={Epoch_Train_SD[k]:.2f}')

                if self.Save_States:
                    self.Save_Checkpoint(ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)

                # Write to TensorBoard
                Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
                Writer_R2_Dict, Writer_ME_Dict, Writer_SD_Dict = {}, {}, {}
                for key, val in Epoch_Train_R2.items():
                    Writer_R2_Dict[f'Train/{key}'] = val
                for key, val in Epoch_Train_ME.items():
                    Writer_ME_Dict[f'Train/{key}'] = val
                for key, val in Epoch_Train_SD.items():
                    Writer_SD_Dict[f'Train/{key}'] = val

                # Testing Phase
                for name, Test in zip(Test_Names, Test_List):
                    Epoch_Labels, Epoch_Preds = [], []
                    for inputs, labels in Test:
                        _, Outputs = self.Test_Batch(inputs, labels)
                        Epoch_Labels.append(labels.cpu().detach().numpy())
                        Epoch_Preds.append(Outputs.cpu().detach().numpy())

                    Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
                    Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)
                    # Epoch_Test_Loss = self.BP_Loss_Fun(
                    #     torch.from_numpy(Epoch_Preds),
                    #     torch.from_numpy(Epoch_Labels)
                    # )
                    
                    Epoch_Test_Loss = self.BP_Loss_Fun(
                        torch.from_numpy(Epoch_Preds).float().to(self.Device),
                        torch.from_numpy(Epoch_Labels).float().to(self.Device)
                    )


                    if Epoch_Preds.ndim == 1 or Epoch_Preds.shape[1] == 1:
                        Epoch_Test_R2 = {'SBP': R2(Epoch_Labels, Epoch_Preds)}
                        Epoch_Test_ME = {'SBP': ME(Epoch_Labels, Epoch_Preds)}
                        Epoch_Test_SD = {'SBP': SD(Epoch_Labels, Epoch_Preds)}
                    else:
                        Epoch_Test_R2 = {
                            'SBP': R2(Epoch_Labels[:, 0], Epoch_Preds[:, 0]),
                            'DBP': R2(Epoch_Labels[:, 1], Epoch_Preds[:, 1])
                        }
                        Epoch_Test_ME = {
                            'SBP': ME(Epoch_Labels[:, 0], Epoch_Preds[:, 0]),
                            'DBP': ME(Epoch_Labels[:, 1], Epoch_Preds[:, 1])
                        }
                        Epoch_Test_SD = {
                            'SBP': SD(Epoch_Labels[:, 0], Epoch_Preds[:, 0]),
                            'DBP': SD(Epoch_Labels[:, 1], Epoch_Preds[:, 1])
                        }

                    print(f'Epoch {name} Loss: {Epoch_Test_Loss:.6e}')
                    for k, v in Epoch_Test_R2.items():
                        print(f'  {name}/{k}: R2={v:.4f} ME={Epoch_Test_ME[k]:.2f} SD={Epoch_Test_SD[k]:.2f}')

                    for key, val in Epoch_Test_R2.items():
                        Writer_R2_Dict[f'{name}/{key}'] = val
                    for key, val in Epoch_Test_ME.items():
                        Writer_ME_Dict[f'{name}/{key}'] = val
                    for key, val in Epoch_Test_SD.items():
                        Writer_SD_Dict[f'{name}/{key}'] = val

                    Writer_Loss_Dict.update({name: Epoch_Test_Loss})

                Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
                Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
                Writer.add_scalars('ME', Writer_ME_Dict, Epoch)
                Writer.add_scalars('SD', Writer_SD_Dict, Epoch)

            except KeyboardInterrupt:
                print(f'Early stopped by interrupt at epoch {Epoch}')
                Interrupt = True
                break

        Writer.close()
        time_elapsed = time.time() - Start_Time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        self.Save_Checkpoint(ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=True)
        if Interrupt:
            raise KeyboardInterrupt


    # ============================================================
    # Training batch
    # ============================================================
    def Train_Batch(self, inputs, BP_labels):
        self.Model_Running.train()
        inputs = inputs.float().to(self.Device)
        BP_labels = BP_labels.float().to(self.Device)
        self.Model_Running.zero_grad()
        BP_outputs = self.Model_Running(inputs)
        BP_loss = self.BP_Loss_Fun(BP_outputs, BP_labels)
        BP_loss.backward()
        self.Optimizer_BP.step()
        return BP_loss.item(), BP_outputs


    # ============================================================
    # Testing batch
    # ============================================================
    def Test_Batch(self, inputs, labels):
        self.Model_Running.eval()
        inputs = inputs.float().to(self.Device)
        labels = labels.float().to(self.Device)
        with torch.no_grad():
            BP_outputs = self.Model_Running(inputs)
            loss = self.BP_Loss_Fun(BP_outputs, labels)
        return loss.item(), BP_outputs


    # ============================================================
    # Checkpoint save
    # ============================================================
    def Save_Checkpoint(self, modelID, timeID, epoch, batchcounter, batchrecordcounter, savemodel=False):
        foldername = modelID
        # if not os.path.isdir(foldername):
        #     os.mkdir(foldername)
        os.makedirs(foldername, exist_ok=True)
        torch.save({
            'model_id': modelID,
            'time_id': timeID,
            'model_state_dict': self.Model_Running.state_dict(),
            'optimizer_state_dict': self.Optimizer_BP.state_dict(),
            'epoch': epoch,
            'batchcounter': batchcounter,
            'batchrecordcounter': batchrecordcounter,
        }, os.path.join(foldername, f'checkpoint_epoch_{epoch}.pth'))
        if savemodel:
            torch.save(self.Model_Running, os.path.join(foldername, 'trained_model.pth'))
