# 필요: pip install torch pillow numpy
import os, random, time, threading
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LETTER_TO_DOTS: Dict[str, Set[int]] = {
    "a": {1},           "b": {1,2},         "c": {1,4},         "d": {1,4,5},       "e": {1,5},
    "f": {1,2,4},       "g": {1,2,4,5},     "h": {1,2,5},       "i": {2,4},         "j": {2,4,5},
    "k": {1,3},         "l": {1,2,3},       "m": {1,3,4},       "n": {1,3,4,5},     "o": {1,3,5},
    "p": {1,2,3,4},     "q": {1,2,3,4,5},   "r": {1,2,3,5},     "s": {2,3,4},       "t": {2,3,4,5},
    "u": {1,3,6},       "v": {1,2,3,6},     "w": {2,4,5,6},     "x": {1,3,4,6},     "y": {1,3,4,5,6},
    "z": {1,3,5,6},
}
CLASSES = list("abcdefghijklmnopqrstuvwxyz")
IDX2CHAR = {i:c for i,c in enumerate(CLASSES)}
CHAR2IDX = {c:i for i,c in enumerate(CLASSES)}

@dataclass
class CellStyle:
    size: int = 64       # 셀 정사각 크기(고정)
    margin: int = 8      # 셀 내부 여백(고정)
    dot_r: int = 7       # 점 반지름(고정)
    blur_sigma: float = 0.0  # 시각만 조금 부드럽게 하고 싶으면 0.1~0.2
    fg: int = 0
    bg: int = 255

def grid_centers(size:int, margin:int)->List[Tuple[float, float]]:
    # 2x3 정확한 그리드(절대 위치) — 항상 동일
    W=H=size
    innerW=W-2*margin; innerH=H-2*margin
    xs=[margin+innerW*0.30, margin+innerW*0.70]
    ys=[margin+innerH*0.20, margin+innerH*0.50, margin+innerH*0.80]
    centers=[]
    for row in range(3):
        for col in range(2):
            centers.append((xs[col], ys[row]))  # 1..6
    return centers

def draw_cell_fixed(dots:Set[int], style:CellStyle)->Image.Image:
    img = Image.new("L", (style.size, style.size), color=style.bg)
    d = ImageDraw.Draw(img)
    centers = grid_centers(style.size, style.margin)
    r = style.dot_r
    for i,(cx,cy) in enumerate(centers, start=1):
        if i not in dots: 
            continue
        bbox = [cx-r, cy-r, cx+r, cy+r]
        d.ellipse(bbox, fill=style.fg)
    if style.blur_sigma>0:
        img = img.filter(ImageFilter.GaussianBlur(style.blur_sigma))
    return img

def compose_word_image(word:str, style:CellStyle, cell_gap:int=12)->Image.Image:
    # 셀 폭/간격 절대 고정 → 예측에서도 같은 값 사용
    cells = [draw_cell_fixed(LETTER_TO_DOTS[ch], style) for ch in word]
    W = len(cells)*style.size + (len(cells)-1)*cell_gap
    out = Image.new("L", (W, style.size), color=style.bg)
    x=0
    for cell in cells:
        out.paste(cell, (x,0))
        x += style.size + cell_gap
    return out

# ------------------------- Dataset (tiny jitter only for training) -------------------------
class SynthDataset(Dataset):
    def __init__(self, n_per_class:int=200, size:int=64, train=True):
        # 학습에는 극소량 랜덤(±1px)만 넣어 경계에 강인성 확보, 표시용은 완전 고정
        self.items=[]
        self.style = CellStyle(size=size, margin=8, dot_r=7, blur_sigma=(0.05 if train else 0.0))
        self.train=train
        for ch in CLASSES:
            for _ in range(n_per_class):
                self.items.append(ch)
        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ch=self.items[idx]
        base = draw_cell_fixed(LETTER_TO_DOTS[ch], self.style)
        # 학습시만 1px 미만의 미세 이동(데이터 다양성)
        img = base
        if self.train:
            shift_x = random.choice([0,0,0,1,-1])
            shift_y = random.choice([0,0,0,1,-1])
            canvas = Image.new("L", (self.style.size, self.style.size), color=self.style.bg)
            canvas.paste(img, (shift_x, shift_y))
            img = canvas
        arr = np.array(img, dtype=np.float32)/255.0
        arr = (arr - 0.5)/0.5
        ten = torch.from_numpy(arr)[None, ...]
        return ten, CHAR2IDX[ch]

# ------------------------- CNN -------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x=self.net(x); x=x.view(x.size(0),-1); return self.fc(x)

def train_quick(save_path="braille_cnn.pt", size=64, epochs=6, batch=128, device=None, log=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if log: log(f"[INFO] device: {device}")
    tr = SynthDataset(n_per_class=250, size=size, train=True)   # 26*250=6500
    va = SynthDataset(n_per_class=60,  size=size, train=False)  # 26*60 =1560
    tr_dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0)
    model=SmallCNN().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    crit=nn.CrossEntropyLoss()
    best=0.0; wait=0
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for x,y in tr_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in va_dl:
                x=x.to(device); y=y.to(device)
                pred=model(x).argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        acc=correct/total
        if log: log(f"[E{ep}] loss={np.mean(losses):.4f}, val_acc={acc*100:.2f}%")
        if acc>best: best=acc; wait=0; torch.save(model.state_dict(), save_path)
        else:
            wait+=1
            if wait>=2: 
                if log: log("[INFO] Early stop."); break
    if log: log(f"[INFO] best_val_acc={best*100:.2f}% -> saved to {save_path}")

def load_or_train(save_path="braille_cnn.pt", log=None):
    device=("cuda" if torch.cuda.is_available() else "cpu")
    model=SmallCNN().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device)); model.eval()
        if log: log(f"[INFO] 모델 로드: {save_path}")
        return model, device
    if log: log("[INFO] 모델 없음 → 빠른 학습 시작")
    train_quick(save_path=save_path, log=log)
    model.load_state_dict(torch.load(save_path, map_location=device)); model.eval()
    return model, device

# ------------------------- Inference (exact slicing) -------------------------
def predict_cell(img:Image.Image, model:nn.Module, device:str)->str:
    arr=np.array(img, dtype=np.float32)/255.0
    arr=(arr-0.5)/0.5
    ten=torch.from_numpy(arr)[None,None,...].to(device)
    with torch.no_grad():
        idx=int(model(ten).argmax(1).item())
    return IDX2CHAR[idx]

def predict_word_image(word_img:Image.Image, model:nn.Module, device:str, *,
                       size:int, gap:int, n_cells:int)->str:
    # 합성 시 쓴 size/gap/n_cells로 정확히 자름 → 순서 뒤섞임 방지
    out=[]
    for i in range(n_cells):
        x0 = i*(size+gap)
        crop = word_img.crop((x0, 0, x0+size, size))
        out.append(predict_cell(crop, model, device))
    return "".join(out)

# ------------------------- Game UI -------------------------
class App:
    def __init__(self, root):
        self.root=root
        self.root.title("Braille: Human vs AI (Fixed)")
        self.root.geometry("920x560")
        self.style_cell = CellStyle(size=64, margin=8, dot_r=7, blur_sigma=0.0)  # 표시용 완전 고정
        self.cell_gap = 14
        self.current_word=""
        self.current_img=None
        self.tk_img=None
        self.round=0; self.total=5
        self.human=0; self.ai=0
        self.ai_pred=""; self.ai_ms=0.0
        self.t0=0.0
        self.model=None; self.device=None
        self.build_ui()
        threading.Thread(target=self.prepare, daemon=True).start()

    def build_ui(self):
        top=ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="인간 vs 인공지능 : 점자 맞히기", font=("Segoe UI", 18, "bold")).pack(side="left")
        self.stat=ttk.Label(top, text="모델 준비 중..."); self.stat.pack(side="right")

        body=ttk.Frame(self.root); body.pack(fill="both", expand=True, padx=12, pady=8)
        left=ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        self.canvas=tk.Label(left, bg="#ffffff", bd=2, relief="groove")
        self.canvas.pack(fill="both", expand=True)

        right=ttk.Frame(body); right.pack(side="right", fill="y")
        ttk.Label(right, text="라운드").pack(anchor="w")
        self.round_lbl=ttk.Label(right, text="0 / 5", font=("Segoe UI", 12)); self.round_lbl.pack(anchor="w", pady=(0,6))

        ttk.Label(right, text="정답(소문자):").pack(anchor="w")
        self.entry=ttk.Entry(right, width=22, font=("Consolas",12)); self.entry.pack(anchor="w", pady=4)
        self.entry.bind("<Return>", lambda e: self.submit())

        btns=ttk.Frame(right); btns.pack(anchor="w", pady=4)
        self.btn_start=ttk.Button(btns, text="라운드 시작", command=self.start, width=12); self.btn_start.grid(row=0, column=0, padx=2)
        self.btn_submit=ttk.Button(btns, text="제출", command=self.submit, state="disabled"); self.btn_submit.grid(row=0, column=1, padx=2)
        self.btn_table=ttk.Button(btns, text="점자 표", command=self.show_table); self.btn_table.grid(row=1, column=0, columnspan=2, pady=6)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(right, text="결과").pack(anchor="w")
        self.txt=tk.Text(right, width=34, height=14, font=("Consolas",10)); self.txt.pack()
        self.txt.configure(state="disabled")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        self.score=ttk.Label(right, text="인간 0 : 0 AI", font=("Segoe UI", 13, "bold")); self.score.pack()

    def log(self, s:str):
        self.stat.config(text=s); self.stat.update_idletasks()

    def prepare(self):
        self.log("모델 준비 중…(없으면 빠르게 학습)")
        self.model, self.device = load_or_train(log=self.append)
        self.log("준비 완료! 라운드를 시작하세요.")

    def append(self, s:str):
        self.txt.configure(state="normal"); self.txt.insert("end", s+"\n"); self.txt.see("end"); self.txt.configure(state="disabled")

    def clear_out(self):
        self.txt.configure(state="normal"); self.txt.delete("1.0","end"); self.txt.configure(state="disabled")

    def rand_word(self, lo=4, hi=7)->str:
        L=random.randint(lo,hi)
        return "".join(random.choice(CLASSES) for _ in range(L))

    def start(self):
        if self.model is None: 
            messagebox.showinfo("잠시만!", "모델 준비 중입니다."); return
        if self.round>=self.total:
            if not messagebox.askyesno("다시 시작", "라운드가 끝났어요. 다시 할까요?"): return
            self.round=0; self.human=0; self.ai=0; self.update_score()
        self.round+=1; self.round_lbl.config(text=f"{self.round} / {self.total}")
        self.current_word = self.rand_word()
        self.current_img = compose_word_image(self.current_word, self.style_cell, cell_gap=self.cell_gap)

        # 표시(보이기 쉽게 2배 리사이즈 — 예측엔 원본 사용)
        show = self.current_img.resize((self.current_img.width*2, self.current_img.height*2), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(show)
        self.canvas.configure(image=self.tk_img)

        os.makedirs("out", exist_ok=True)
        self.current_img.save(f"out/round_{self.round}_{self.current_word}.png")

        self.entry.delete(0, tk.END); self.entry.focus_set()
        self.btn_submit.config(state="normal"); self.btn_start.config(state="disabled")
        self.clear_out(); self.append(f"[Round {self.round}] 사람이 먼저 맞혀보세요!")

        # 타이머
        self.t0=time.perf_counter()
        # AI 미리 계산(고정 자르기: n_cells=len(word))
        def infer():
            t=time.perf_counter()
            pred = predict_word_image(self.current_img, self.model, self.device,
                                      size=self.style_cell.size, gap=self.cell_gap,
                                      n_cells=len(self.current_word))
            dt=(time.perf_counter()-t)*1000.0
            self.ai_pred=pred; self.ai_ms=dt
        threading.Thread(target=infer, daemon=True).start()

    def submit(self):
        if self.btn_submit['state']=="disabled": return
        human_ans=self.entry.get().strip().lower()
        if not human_ans: messagebox.showinfo("입력 필요", "정답을 입력하세요."); return
        human_ms=(time.perf_counter()-self.t0)*1000.0

        # AI 결과 대기(아주 짧음)
        wait=0
        while self.ai_pred=="" and wait<2000:
            time.sleep(0.01); wait+=10

        gt=self.current_word
        h_ok=(human_ans==gt); a_ok=(self.ai_pred==gt)

        if h_ok and (not a_ok or human_ms < self.ai_ms): self.human+=1; result="인간 승"
        elif a_ok and (not h_ok or self.ai_ms <= human_ms): self.ai+=1; result="AI 승"
        else: result="무승부"
        self.update_score()

        self.append(
            f"정답: {gt}\n"
            f"인간: {human_ans} | {'정답' if h_ok else '오답'} | {human_ms:.0f} ms\n"
            f"AI  : {self.ai_pred} | {'정답' if a_ok else '오답'} | {self.ai_ms:.0f} ms\n"
            f"결과: {result}\n" + "-"*34
        )

        self.btn_submit.config(state="disabled"); self.btn_start.config(state="normal")
        if self.round>=self.total:
            final = "🎉 인간 승리!" if self.human>self.ai else ("🤖 AI 승리!" if self.ai>self.human else "⚖️ 무승부!")
            messagebox.showinfo("경기 종료", f"{final}\n최종 스코어  인간 {self.human} : {self.ai} AI")
        self.ai_pred=""; self.ai_ms=0.0

    def update_score(self):
        self.score.config(text=f"인간 {self.human} : {self.ai} AI")

    def show_table(self):
        win=tk.Toplevel(self.root); win.title("영어 점자 표"); win.geometry("520x520")
        box=tk.Text(win, font=("Consolas",12)); box.pack(fill="both", expand=True)
        def dots_to_unicode(dots:Set[int])->str:
            mask=0
            for i in dots: mask|=(1<<(i-1))
            return chr(0x2800+mask)
        lines=[]
        for ch in CLASSES:
            u=dots_to_unicode(LETTER_TO_DOTS[ch])
            d="".join(str(i) for i in sorted(LETTER_TO_DOTS[ch]))
            lines.append(f"{ch} : {u} (dots {d})")
        mid=(len(lines)+1)//2
        left,right=lines[:mid],lines[mid:]
        for i in range(mid):
            l=left[i] if i<len(left) else ""
            r=right[i] if i<len(right) else ""
            box.insert("end", f"{l:<26}   {r}\n")
        box.configure(state="disabled")

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root=tk.Tk(); App(root); root.mainloop()

if __name__=="__main__":
    main()
