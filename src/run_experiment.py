import argparse, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASSES=["akiec","bcc","bkl","df","mel","nv","vasc"]
C2I={c:i for i,c in enumerate(CLASSES)}

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

class DS(Dataset):
    def __init__(self,df,imgdir,train=False):
        self.df=df.reset_index(drop=True); self.imgdir=Path(imgdir)
        t=[transforms.Resize((224,224))]
        if train:
            t += [transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.5),
                  transforms.RandomRotation(15)]
        t += [transforms.ToTensor(),
              transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.t=transforms.Compose(t)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        x=Image.open(self.imgdir/(str(r.image_id)+".jpg")).convert("RGB")
        return self.t(x),C2I[str(r.dx)]

def make_model(name):
    if name=="mobilenet_v2":
        m=models.mobilenet_v2(weights="DEFAULT")
        m.classifier[1]=nn.Linear(m.classifier[1].in_features,7)
    elif name=="efficientnet_b0":
        m=models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1]=nn.Linear(m.classifier[1].in_features,7)
    else:
        m=models.resnet50(weights="DEFAULT")
        m.fc=nn.Linear(m.fc.in_features,7)
    return m

def evaluate(m,loader,dev):
    m.eval(); y=[]; p=[]
    with torch.no_grad():
        for x,t in loader:
            q=m(x.to(dev)).argmax(1).cpu().numpy()
            p.extend(q); y.extend(t.numpy())
    return accuracy_score(y,p),f1_score(y,p,average="macro")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="mobilenet_v2",
                    choices=["mobilenet_v2","efficientnet_b0","resnet50","all"])
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    seed()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",dev)

    s=pd.read_csv("data/splits/ham10000_splits.csv")
    img="data/raw/ham10000/images"

    names=["mobilenet_v2","efficientnet_b0","resnet50"] if args.model=="all" else [args.model]

    for name in names:
        print("\n"+"="*50)
        print("MODEL:",name)
        print("="*50)

        tr=s[s.split=="train"].copy()
        va=s[s.split=="val"].copy()
        te=s[s.split=="test"].copy()

        epochs=args.epochs
        if args.smoke:
            tr=tr.head(64); va=va.head(32); te=te.head(32); epochs=1

        tl=DataLoader(DS(tr,img,True),batch_size=16,shuffle=True,num_workers=0)
        vl=DataLoader(DS(va,img),batch_size=16,num_workers=0)
        tel=DataLoader(DS(te,img),batch_size=16,num_workers=0)

        m=make_model(name).to(dev)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
        lossfn=nn.CrossEntropyLoss()

        best=-1
        for e in range(epochs):
            m.train(); total=0
            for x,y in tl:
                x,y=x.to(dev),y.to(dev)
                opt.zero_grad()
                loss=lossfn(m(x),y)
                loss.backward(); opt.step()
                total+=loss.item()

            a,f=evaluate(m,vl,dev)
            print(f"Epoch {e+1}/{epochs} loss={total/len(tl):.4f} val_acc={a:.4f} val_f1={f:.4f}")

            if f>best:
                best=f
                Path("results/models").mkdir(parents=True,exist_ok=True)
                torch.save(m.state_dict(),f"results/models/{name}_best.pt")

        a,f=evaluate(m,tel,dev)
        print(f"TEST accuracy={a:.4f} macro_F1={f:.4f}")

        Path("results/metrics").mkdir(parents=True,exist_ok=True)
        with open(f"results/metrics/{name}_ham10000_test.json","w") as z:
            json.dump({"accuracy":float(a),"macro_f1":float(f)},z,indent=2)

if __name__=="__main__":
    main()
