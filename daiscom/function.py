import sys
import time
import numpy as np
import pandas as pd
import argparse
import random
from random import randint, sample
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

# function
# select gene
def daiscom_preprocessing(refexp,target_sample,marker,outdir,platform):
     
    if platform =='Rc':
        pure_dir = "../data/example_pure/rna_counts/"
        B_CRNA = pd.read_csv(pure_dir+'B_RNA_cpm.txt', index_col=0,sep='\t')#213
        NK_CRNA = pd.read_csv(pure_dir+'NK_RNA_cpm.txt', index_col=0,sep='\t')#153
        CD8_CRNA = pd.read_csv(pure_dir+'CD8_RNA_cpm.txt', index_col=0,sep='\t')#227
        CD4_CRNA = pd.read_csv(pure_dir+'CD4_RNA_cpm.txt', index_col=0,sep='\t')#291
        NEU_CRNA = pd.read_csv(pure_dir+'Neu_RNA_cpm.txt', index_col=0,sep='\t')#91
        MONO_CRNA = pd.read_csv(pure_dir+'MONO_RNA_cpm.txt', index_col=0,sep='\t')#327
        FIB_CRNA = pd.read_csv(pure_dir+'FIB_RNA_cpm.txt', index_col=0, sep='\t') #132
        END_CRNA = pd.read_csv(pure_dir+'end_RNA_cpm.txt', index_col=0, sep='\t') #99

        commongene = list(set(list(marker["0"])).intersection(list(refexp.index))
                          .intersection(list(target_sample.index)).intersection(list(B_CRNA.index)))
        commongene.sort()
        
        B = B_CRNA.reindex(commongene)
        CD4 = CD4_CRNA.reindex(commongene)
        CD8 = CD8_CRNA.reindex(commongene)
        MONO = MONO_CRNA.reindex(commongene)
        NEU = NEU_CRNA.reindex(commongene)
        FIB = FIB_CRNA.reindex(commongene)
        END = END_CRNA.reindex(commongene)
        NK = NK_CRNA.reindex(commongene)
        
    elif platform =='Rt':
        pure_dir = "../data/example_pure/rna_tpm/"
        B_TRNA = pd.read_csv(pure_dir+'B_RNA_TPM.txt', index_col=0,sep='\t')#213
        NK_TRNA = pd.read_csv(pure_dir+'NK_RNA_TPM.txt', index_col=0,sep='\t')#153
        CD8_TRNA = pd.read_csv(pure_dir+'CD8_RNA_TPM.txt', index_col=0,sep='\t')#227
        CD4_TRNA = pd.read_csv(pure_dir+'CD4_RNA_TPM.txt', index_col=0,sep='\t')#291
        NEU_TRNA = pd.read_csv(pure_dir+'Neu_RNA_TPM.txt', index_col=0,sep='\t')#91
        MONO_TRNA = pd.read_csv(pure_dir+'MONO_RNA_TPM.txt', index_col=0,sep='\t')#327
        FIB_TRNA = pd.read_csv(pure_dir+'FIB_RNA_TPM.txt', index_col=0, sep='\t') #132
        END_TRNA = pd.read_csv(pure_dir+'end_RNA_TPM.txt', index_col=0, sep='\t') #99
        
        commongene = list(set(list(marker["0"])).intersection(list(refexp.index))
                          .intersection(list(target_sample.index)).intersection(list(B_TRNA.index)))
        commongene.sort()
        
        B = B_TRNA.reindex(commongene)
        CD4 = CD4_TRNA.reindex(commongene)
        CD8 = CD8_TRNA.reindex(commongene)
        MONO = MONO_TRNA.reindex(commongene)
        NEU = Neu_TRNA.reindex(commongene)
        FIB = FIB_TRNA.reindex(commongene)
        END = END_TRNA.reindex(commongene)
        NK = NK_TRNA.reindex(commongene)
        
    elif platform =='M':
        pure_dir = "../data/example_pure/microarray/"       
        B_array = pd.read_csv(pure_dir+'exp_Bcell_norm_gene.txt', index_col=0,sep='\t')#128
        CD4_array = pd.read_csv(pure_dir+'exp_cd4_norm_gene.txt', index_col=0,sep='\t')#184
        CD8_array = pd.read_csv(pure_dir+'exp_cd8_norm_gene.txt', index_col=0,sep='\t')#64
        MONO_array = pd.read_csv(pure_dir+'exp_Monocytes_norm_gene.txt', index_col=0,sep='\t')#116
        NEU_array = pd.read_csv(pure_dir+'exp_Neutrophils_norm_gene.txt', index_col=0,sep='\t')#18
        FIB_array = pd.read_csv(pure_dir+'exp_Fibroblast_norm_gene.txt', index_col=0,sep='\t') #45
        END_array = pd.read_csv(pure_dir+'exp_Endothelium_norm_gene.txt', index_col=0, sep='\t') #116
        NK_array = pd.read_csv(pure_dir+'exp_NKcell_norm_gene.txt', index_col=0, sep='\t') #15
        
        commongene = list(set(list(marker["0"])).intersection(list(refexp.index))
                          .intersection(list(target_sample.index)).intersection(list(B_array.index)))
        commongene.sort()

        B = B_array.reindex(commongene)
        CD4 = CD4_array.reindex(commongene)
        CD8 = CD8_array.reindex(commongene)
        MONO = MONO_array.reindex(commongene)
        NEU = Neu_array.reindex(commongene)
        FIB = FIB_array.reindex(commongene)
        END = END_array.reindex(commongene)
        NK = NK_array.reindex(commongene)
        
        B = 2**B
        MONO = 2**MONO
        CD8 = 2**CD8
        CD4 = 2**CD4
        NEU = 2**Neu
        FIB = 2**FIB
        END = 2**END
        NK = 2**NK
        
    else:
        print("ERROR: unknown platform.")
        sys.exit(1)
        
    trainsam = refexp.reindex(commongene)
    testsam = target_sample.reindex(commongene)
    C_B = B.T.values
    C_NK = NK.T.values
    C_CD4 = CD4.T.values
    C_CD8 = CD8.T.values
    C_MONO= MONO.T.values
    C_FIB=FIB.T.values
    C_END=END.T.values
    C_NEU=NEU.T.values
    
    pd.DataFrame(commongene).to_csv(outdir+"DAISCOM_commongene.txt",sep="\t",index=False)
    return(trainsam,testsam,commongene,C_B,C_NK,C_CD4,C_CD8,C_MONO,C_FIB,C_END,C_NEU)

#进度条显示
class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' +'('+'%d' % self.i+'/'+'%d' %self.max_steps+')'+ '\r' 
        #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

# data argument
def daiscom_simulation(trainsam, trainfra,C_B,C_NK,C_CD4,C_CD8,C_MONO,C_FIB,C_END,C_NEU, random_seed, n, outdir, min_f=0.01, max_f=0.99):
    gn = trainsam.shape[0]
    cn = trainfra.shape[0]
    sn = trainfra.shape[1]
    N = sn*n
    random.seed(random_seed)
    np.random.seed(random_seed)

    process_bar = ShowProcess(N, 'mixture_simulation finish!')

    mixsam = np.zeros(shape=(N, gn))
    mixfra = np.zeros(shape=(N, cn))
    
    for i,sampleid in enumerate(trainfra.columns):

        for j in range(n):
            
            mix_fraction =round(random.uniform(min_f,max_f),8)
            fraction = np.random.dirichlet([1]*cn,1)*(1-mix_fraction)
            mixsam[i*n+j] = np.array(trainsam.T.values[i])*mix_fraction
            #print("aa",trainsam.T.values[i])
            mixfra[i*n+j] = trainfra.T.values[i]*mix_fraction

            for k, celltype in enumerate(trainfra[sampleid].T.index):

                if celltype == "B.cells":
                    pure = C_B[random.randint(0,C_B.shape[0]-1)]
                elif celltype == "CD4.T.cells":
                    pure = C_CD4[random.randint(0,C_CD4.shape[0]-1)]
                elif celltype == "CD8.T.cells":
                    pure = C_CD8[random.randint(0,C_CD8.shape[0]-1)]
                elif celltype == "NK.cells":
                    pure = C_NK[random.randint(0,C_NK.shape[0]-1)]
                elif celltype == "monocytic.lineage":
                    pure = C_MONO[random.randint(0,C_MONO.shape[0]-1)]
                elif celltype == "neutrophils":
                    pure = C_NEU[random.randint(0,C_NEU.shape[0]-1)]
                elif celltype == "fibroblasts":
                    pure = C_FIB[random.randint(0,C_FIB.shape[0]-1)]
                elif celltype == "endothelial.cells":
                    pure = C_END[random.randint(0,C_END.shape[0]-1)]
                else:
                    print("ERROR: unrecognized cell type.")
                    sys.exit(1)

                mixsam[i*n+j] += np.array(pure)*fraction[0,k]
                #print("bb",mixsam[i*n+j])
                mixfra[i*n+j][k] += fraction[0,k]
                    
            process_bar.show_process()
            time.sleep(0.0001)
    
    mixsam = pd.DataFrame(mixsam.T,index = trainsam.index)
    mixfra = pd.DataFrame(mixfra.T,index = trainfra.index)
    celltypes = list(mixfra.index)
    mixsam.to_csv(outdir+"DAISCOM_mixsam.txt",sep="\t")
    mixfra.to_csv(outdir+"DAISCOM_mixfra.txt",sep="\t")
    pd.DataFrame(celltypes).to_csv(outdir+"DAISCOM_celltypes.txt",sep="\t")
    
    return (mixsam, mixfra, celltypes)

def minmaxscaler(x):
    x = np.log2(x + 1)
    x = (x - x.min(axis = 0))/(x.max(axis = 0) - x.min(axis = 0))
    return x

def Dataloader(xtr,ytr,batchsize):
    """
    :ytr - fraction cell type*sample

    :batchsize 
    """
    train_dataset = Data.TensorDataset(xtr, ytr)
    trainloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size = batchsize,
        shuffle=True,
        num_workers=0    # set multi-work num read data
    )
    return trainloader

class train_preprocessing():
    def __init__(self, tx, ty, ts, rs):

        tx = minmaxscaler(tx).values
        ty = ty.values
        xtr, xve, ytr, yve = train_test_split(tx.T, ty.T, test_size=ts, random_state=rs)

        self.xtr = torch.from_numpy(xtr)
        self.xve = torch.from_numpy(xve)
        self.ytr = torch.from_numpy(ytr)
        self.yve = torch.from_numpy(yve)
        
        if torch.cuda.is_available():
            self.xtr = self.xtr.cuda(0)
            self.xve = self.xve.cuda(0) 
            self.ytr = self.ytr.cuda(0)
            self.yve = self.yve.cuda(0) 
            
class MLP(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP, self).__init__()  
        L1 = 256
        L2 = 512
        L3 = 128
        L4 = 32
        L5 = 16
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.BatchNorm1d(L2),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.Tanh(),
            nn.Linear(L3,L4),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
            # nn.Softmax()
        )
    def forward(self, x):   
        y = self.hidden(x)    
        y = self.predict(y)   
        return y
    
def evaluate(model,xve,yve,epoch):

    model.eval()
    vout = model(xve)
    ve_p = Variable(vout,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    ve_y = Variable(yve,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    res = np.abs(ve_p-ve_y)
    mae_ve = np.mean(res)
    return mae_ve

def daiscom_training(mixsam,mixfra,random_seed,outdir,num_epoches=300,lr=1e-4,batchsize=64):
    
    # Fixed parameter definition
    lr_min = 1e-5   # Minimum learning rate
    de_lr = 0.9    # Attenuation index
    mae_tr_prev = 0.05   # training mae initial value
    dm = 0   # Attenuation threshold
    mae_ve = []
    min_mae = 1
    n = 0
    min_epoch = 20 
    cn = mixfra.shape[0]
    gn = mixsam.shape[0]
    # Data preprocessing
    data = train_preprocessing(tx = mixsam,ty = mixfra, ts = 0.1, rs = random_seed)
    trainloader = Dataloader(xtr=data.xtr, ytr=data.ytr, batchsize=batchsize)
    
    # Model definition
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    model = MLP(INPUT_SIZE = gn,OUTPUT_SIZE = cn).double()
    if torch.cuda.is_available():
        model = model.cuda(0)    
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)  
    loss_func = torch.nn.MSELoss()     
    
    # training
    for epoch in range(num_epoches):
        
        mae_tr=[]
        for step, (batch_x, batch_y) in enumerate(trainloader):

            model.train()
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_func(out, batch_y) 
            loss.backward() 
            optimizer.step() 
            tr_p = Variable(out,requires_grad = False).cpu().numpy().reshape(batchsize*cn)  
            tr_t = Variable(batch_y,requires_grad = False).cpu().numpy().reshape(batchsize*cn)
            mae_tr.append(np.mean(abs(tr_p - tr_t)))
        
        #print('Epoch {}/{},MSEloss:{:.4f}'.format(epoch, num_epoches, loss.item()))
        mae_tr_change = (np.mean(mae_tr)-mae_tr_prev)
        mae_tr_prev = np.mean(mae_tr)
        if mae_tr_change > dm:         
            optimizer.param_groups[0]['lr'] *= de_lr   
        if optimizer.param_groups[0]['lr'] < lr_min:
            optimizer.param_groups[0]['lr'] = lr_min

        mae_ve.append(evaluate(model,data.xve,data.yve,epoch))        
        if epoch >= min_epoch:
            if mae_ve[epoch] <= min_mae:
                min_mae = mae_ve[epoch]
                torch.save(model.state_dict(), outdir+'DAISCOM_model.pkl')
                n = 0
            else:
                n += 1
            if n==10:
                break

    model.load_state_dict(torch.load(outdir+'DAISCOM_model.pkl'))
    print("model_trainging finish!")
    return model

def daisocm_prediction(model, testsam, celltypes, commongene,outdir):
    
    data = testsam.reindex(commongene)
    data = minmaxscaler(testsam).values.T
    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        data = data.cuda(0)
        
    model.eval()
    out = model(data)
    
    pred = Variable(out,requires_grad=False).cpu().numpy().reshape(testsam.shape[1],len(celltypes))    
    pred_result = pd.DataFrame(pred.T,index=celltypes,columns=testsam.columns)
    
    pred_result.to_csv(outdir+"DAISCOM_result.txt",sep="\t")
    print("result_prediction finish!")
    return pred_result

def model_load(commongene, celltypes, modelpath, outdir, random_seed):
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    model = MLP(INPUT_SIZE = len(commongene),OUTPUT_SIZE = len(celltypes)).double()
    if torch.cuda.is_available():
        model = model.cuda(0)
        model.load_state_dict(torch.load(modelpath))
    else:
        model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        
    return model
