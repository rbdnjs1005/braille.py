# braille_ai_game.py
# ------------------------------------------------------------
# 인간 vs AI 점자 맞히기 게임 (GUI)
# 1) 필요한 패키지 설치:  pip install torch pillow numpy
# 2) 실행: python braille_ai_game.py
#  - 첫 실행 시 합성 데이터로 간단 학습 후 시작(빠르게 끝나도록 설정)
#  - 라운드마다: 점자 이미지 제시 → 사람 답 입력 → AI 답/속도 공개 → 점수 갱신
#  - "점자 표 보기" 버튼으로 알파벳-점자 매핑 확인
# ------------------------------------------------------------
import os, random, time, threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# ---- torch (AI) ----
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 점자 매핑 (영어 Grade-1: a~z) =====================
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

# ===================== 합성 이미지 스타일 =====================
@dataclass
class CellStyle:
    size: int = 64
    margin: int = 8
    base_r: int = 7
    jitter: float = 1.6
    blur_sigma: float = 0.5
    rot_deg: float = 5.0
    fg: int = 0
    bg: int = 255
    draw_empty: bool = False  # 비활성 점도 표시할지 (훈련/디버그용)

def _grid_centers(size:int, margin:int) -> List[Tuple[float,float]]:
    W=H=size
    innerW=W-2*margin; innerH=H-2*margin
    xs=[margin+innerW*0.30, margin+innerW*0.70]  # 좌/우
    ys=[margin+innerH*0.20, margin+innerH*0.50, margin+innerH*0.80]  # 상/중/하
    centers=[]
    for row in range(3):
        for col in range(2):
            centers.append((xs[col], ys[row]))  # 순서: 1..6
    return centers

def draw_braille_cell(dots:Set[int], style:CellStyle) -> Image.Image:
    img = Image.new("L", (style.size, style.size), color=style.bg)
    draw = ImageDraw.Draw(img)
    centers = _grid_centers(style.size, style.margin)
    for i,(cx,cy) in enumerate(centers, start=1):
        present = i in dots
        if not present and not style.draw_empty:
            continue
        jx = random.uniform(-style.jitter, style.jitter)
        jy = random.uniform(-style.jitter, style.jitter)
        r  = max(3, int(style.base_r + random.uniform(-2,2)))
        bbox = [cx+jx-r, cy+jy-r, cx+jx+r, cy+jy+r]
        fill = style.fg if present else 180
        draw.ellipse(bbox, fill=fill)
    if style.rot_deg>0:
        img = img.rotate(random.uniform(-style.rot_deg, style.rot_deg), resample=Image.BICUBIC, fillcolor=style.bg)
    if style.blur_sigma>0:
        img = img.filter(ImageFilter.GaussianBlur(style.blur_sigma))
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 4.0, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def compose_word_image(word:str, style:CellStyle, cell_gap:int=12) -> Image.Image:
    cells=[]
    for ch in word:
        if ch==" ":
            cells.append(Image.new("L", (style.size,style.size), color=style.bg))
        else:
            cells.append(draw_braille_cell(LETTER_TO_DOTS[ch], style))
    W = len(cells)*style.size + (len(cells)-1)*cell_gap
    H = style.size
    out = Image.new("L",(W,H),color=style.bg)
    x=0
    for cell in cells:
        out.paste(cell,(x,0))
        x += style.size + cell_gap
    return out

# ===================== 합성 데이터셋 & 간단 CNN =====================
class BrailleSynthDataset(Dataset):
    def __init__(self, n_per_class:int=100, size:int=64, train:bool=True):
        self.items=[]
        self.train=train
        if train:
            self.style = CellStyle(size=size, jitter=2.0, blur_sigma=0.7, rot_deg=6.0)
        else:
            self.style = CellStyle(size=size, jitter=0.8, blur_sigma=0.3, rot_deg=3.0)
        for ch in CLASSES:
            for _ in range(n_per_class):
                self.items.append(ch)
        random.shuffle(self.items)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ch=self.items[idx]
        img=draw_braille_cell(LETTER_TO_DOTS[ch], self.style)
        arr=np.array(img, dtype=np.float32)/255.0
        arr=(arr-0.5)/0.5
        ten=torch.from_numpy(arr)[None,...]
        label=CHAR2IDX[ch]
        return ten, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 32x32
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16x16
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8x8
            nn.Conv2d(64,96,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc=nn.Linear(96,num_classes)
    def forward(self,x):
        x=self.net(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

def train_quick(save_path="braille_cnn.pt", size=64, epochs=4, batch=128, device=None, log_cb=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    if log_cb: log_cb(f"[INFO] device: {device}")
    train_ds = BrailleSynthDataset(n_per_class=100, size=size, train=True)   # 26*100=2600
    val_ds   = BrailleSynthDataset(n_per_class=30,  size=size, train=False)  # 26*30=780
    train_dl=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=0)
    val_dl  =DataLoader(val_ds,batch_size=batch,shuffle=False,num_workers=0)

    model=SmallCNN().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    crit=nn.CrossEntropyLoss()
    best=0.0; patience=2; hits=0
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for x,y in train_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(); logit=model(x); loss=crit(logit,y)
            loss.backward(); opt.step(); losses.append(loss.item())
        # val
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in val_dl:
                x=x.to(device); y=y.to(device)
                pred=model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total
        if log_cb: log_cb(f"[E{ep}] loss={np.mean(losses):.4f}, val_acc={acc*100:.2f}%")
        if acc>best:
            best=acc; hits=0
            torch.save(model.state_dict(), save_path)
        else:
            hits+=1
            if hits>=patience:
                if log_cb: log_cb("[INFO] Early stop.")
                break
    if log_cb: log_cb(f"[INFO] best_val_acc={best*100:.2f}% -> saved to {save_path}")

def load_or_train(save_path="braille_cnn.pt", log_cb=None):
    device=("cuda" if torch.cuda.is_available() else "cpu")
    model=SmallCNN().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        if log_cb: log_cb(f"[INFO] 모델 로드: {save_path}")
        return model, device
    # 빠른 학습
    if log_cb: log_cb("[INFO] 모델이 없어 빠르게 학습을 시작합니다(수십초 내).")
    train_quick(save_path=save_path, log_cb=log_cb)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model, device

# ===================== 추론 유틸 =====================
def predict_cell(img:Image.Image, model:nn.Module, device:str) -> str:
    arr=np.array(img, dtype=np.float32)/255.0
    arr=(arr-0.5)/0.5
    ten=torch.from_numpy(arr)[None,None,...].to(device)
    with torch.no_grad():
        idx=int(model(ten).argmax(1).item())
    return IDX2CHAR[idx]

def predict_word_image(word_img:Image.Image, model:nn.Module, device:str, size:int=64, gap:int=12) -> str:
    W,H=word_img.size
    n = (W + gap)//(size+gap)
    out=[]
    for i in range(n):
        x0 = i*(size+gap)
        crop = word_img.crop((x0,0,x0+size,size))
        out.append(predict_cell(crop, model, device))
    return "".join(out)

# ===================== 게임 로직 (Tkinter) =====================
class BrailleGameApp:
    def __init__(self, root):
        self.root=root
        root.title("Braille: Human vs AI")
        root.geometry("920x560")
        root.configure(bg="#fafafa")

        # 상태
        self.model=None
        self.device=None
        self.cell_size=64
        self.cell_gap=12
        self.style = CellStyle(size=self.cell_size, jitter=0.8, blur_sigma=0.35, rot_deg=2.0)
        self.current_word=""
        self.current_img=None
        self.tk_img=None
        self.round_idx=0
        self.total_rounds=5
        self.score_human=0
        self.score_ai=0
        self.human_start_time=0.0
        self.ai_guess=""
        self.ai_time_ms=0.0

        # UI 레이아웃
        self.build_ui()

        # 모델 로드/학습(백그라운드 스레드)
        threading.Thread(target=self.prepare_model, daemon=True).start()

    def build_ui(self):
        # 상단
        top=ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=8)
        self.title_lbl=ttk.Label(top, text="인간 vs 인공지능 : 점자 맞히기", font=("Segoe UI", 18, "bold"))
        self.title_lbl.pack(side="left")

        self.status_lbl=ttk.Label(top, text="모델 준비 중...", font=("Segoe UI", 10))
        self.status_lbl.pack(side="right")

        # 본문
        body=ttk.Frame(self.root); body.pack(fill="both", expand=True, padx=12, pady=8)

        # 좌: 이미지 영역
        left=ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        self.canvas=tk.Label(left, bg="#ffffff", relief="groove", bd=2)
        self.canvas.pack(fill="both", expand=True)

        # 우: 컨트롤
        right=ttk.Frame(body); right.pack(side="right", fill="y")
        ttk.Label(right, text="라운드", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0,4))
        self.round_lbl=ttk.Label(right, text="0 / 5", font=("Segoe UI", 12))
        self.round_lbl.pack(anchor="w", pady=(0,8))

        ttk.Label(right, text="당신의 정답(소문자):").pack(anchor="w")
        self.entry=ttk.Entry(right, width=24, font=("Consolas", 12))
        self.entry.pack(anchor="w", pady=4)
        self.entry.bind("<Return>", lambda e: self.submit())

        btns=ttk.Frame(right); btns.pack(anchor="w", pady=6)
        self.btn_start=ttk.Button(btns, text="라운드 시작", command=self.start_round, width=12)
        self.btn_start.grid(row=0, column=0, padx=2)
        self.btn_submit=ttk.Button(btns, text="제출", command=self.submit, width=8, state="disabled")
        self.btn_submit.grid(row=0, column=1, padx=2)
        self.btn_table=ttk.Button(btns, text="점자 표 보기", command=self.show_table)
        self.btn_table.grid(row=1, column=0, columnspan=2, pady=(6,0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(right, text="결과", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.result_txt=tk.Text(right, width=34, height=14, font=("Consolas", 10))
        self.result_txt.pack(pady=4)
        self.result_txt.configure(state="disabled")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        self.score_lbl=ttk.Label(right, text="인간 0 : 0 AI", font=("Segoe UI", 13, "bold"))
        self.score_lbl.pack(anchor="center")

        # 하단
        bottom=ttk.Frame(self.root); bottom.pack(fill="x", padx=12, pady=(0,10))
        self.save_hint=ttk.Label(bottom, text="생성 이미지: ./out/round_#.png 로 저장됩니다.", font=("Segoe UI", 9))
        self.save_hint.pack(side="left")

        # 스타일 다듬기
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass

    def log(self, msg:str):
        self.status_lbl.config(text=msg)
        self.status_lbl.update_idletasks()

    def prepare_model(self):
        self.log("모델 준비 중… (없으면 빠르게 학습)")
        self.model, self.device = load_or_train(save_path="braille_cnn.pt", log_cb=self.append_result_console)
        self.log("모델 준비 완료! 라운드를 시작하세요.")

    def rand_word(self, lo=3, hi=6) -> str:
        L=random.randint(lo,hi)
        return "".join(random.choice(CLASSES) for _ in range(L))

    def start_round(self):
        if self.model is None:
            messagebox.showinfo("잠시만!", "모델 준비가 아직 끝나지 않았습니다.")
            return
        if self.round_idx >= self.total_rounds:
            if not messagebox.askyesno("게임 종료", "모든 라운드가 끝났어요. 다시 시작할까요?"):
                return
            self.round_idx=0
            self.score_human=0
            self.score_ai=0
            self.update_score()

        self.round_idx += 1
        self.round_lbl.config(text=f"{self.round_idx} / {self.total_rounds}")

        # 문제 생성
        self.current_word = self.rand_word(4,7)
        img = compose_word_image(self.current_word, self.style, cell_gap=self.cell_gap)
        self.current_img = img

        # 이미지 표시
        show_img = img.resize((img.width*2, img.height*2), Image.NEAREST)  # 보기 쉽게 2배
        self.tk_img = ImageTk.PhotoImage(show_img)
        self.canvas.configure(image=self.tk_img)

        # 저장
        os.makedirs("out", exist_ok=True)
        img.save(f"out/round_{self.round_idx}_{self.current_word}.png")

        # 입력 준비
        self.entry.delete(0, tk.END)
        self.entry.focus_set()
        self.btn_submit.config(state="normal")
        self.btn_start.config(state="disabled")

        # 사람 타이머 시작
        self.human_start_time = time.perf_counter()

        # AI 추론(미리 계산하되, 결과는 제출 후 공개)
        def infer_ai():
            t0=time.perf_counter()
            pred = predict_word_image(self.current_img, self.model, self.device, size=self.cell_size, gap=self.cell_gap)
            t1=time.perf_counter()
            self.ai_guess = pred
            self.ai_time_ms = (t1-t0)*1000.0
        threading.Thread(target=infer_ai, daemon=True).start()

        # 결과창 메시지
        self.clear_result_console()
        self.append_result_console(f"[Round {self.round_idx}] 문제 생성! 사람이 먼저 맞혀보세요.\n")

    def submit(self):
        if self.btn_submit['state']=="disabled":
            return
        human_ans = self.entry.get().strip().lower()
        if not human_ans:
            messagebox.showinfo("입력 필요", "정답을 입력해 주세요.")
            return

        human_time_ms = (time.perf_counter() - self.human_start_time)*1000.0
        gt = self.current_word

        # AI가 아직 계산 중이면 살짝 대기 (보통 매우 빠름)
        wait_ms=0
        while self.ai_guess=="" and wait_ms<3000:
            time.sleep(0.02); wait_ms+=20

        # 채점
        human_correct = (human_ans==gt)
        ai_correct    = (self.ai_guess==gt)

        if human_correct and (not ai_correct or human_time_ms < self.ai_time_ms):
            self.score_human += 1
            winner="인간 승"
        elif ai_correct and (not human_correct or self.ai_time_ms <= human_time_ms):
            self.score_ai += 1
            winner="AI 승"
        else:
            # 둘다 오답인 경우 무승부
            winner="무승부"

        self.update_score()
        self.append_result_console(
            f"정답( GT ): {gt}\n"
            f"인간   : {human_ans}   | {'정답' if human_correct else '오답'} | {human_time_ms:.0f} ms\n"
            f"AI     : {self.ai_guess if self.ai_guess else '(계산중)'}   | {'정답' if ai_correct else '오답'} | {self.ai_time_ms:.0f} ms\n"
            f"결과   : {winner}\n"
            "----------------------------------------\n"
        )

        # 다음 진행
        self.btn_submit.config(state="disabled")
        self.btn_start.config(state="normal")

        if self.round_idx>=self.total_rounds:
            if   self.score_human > self.score_ai: final="🎉 인간 승리!"
            elif self.score_ai > self.score_human: final="🤖 AI 승리!"
            else: final="⚖️ 무승부!"
            messagebox.showinfo("경기 종료", f"{final}\n최종 스코어  인간 {self.score_human} : {self.score_ai} AI")

        # 상태 초기화
        self.ai_guess=""; self.ai_time_ms=0.0

    def update_score(self):
        self.score_lbl.config(text=f"인간 {self.score_human} : {self.score_ai} AI")

    def show_table(self):
        win = tk.Toplevel(self.root)
        win.title("영어 점자 표 (Grade-1)")
        win.geometry("520x520")
        txt=tk.Text(win, font=("Consolas", 12))
        txt.pack(fill="both", expand=True)
        txt.insert("end", "영어 알파벳 → 점자(유니코드) / 점 번호\n\n")
        def to_unicode(dots:Set[int])->str:
            # 6점 유니코드 기본 블록 조합
            mask=0
            for i in dots:
                mask |= (1<<(i-1))
            return chr(0x2800+mask)
        lines=[]
        for ch in CLASSES:
            dots=LETTER_TO_DOTS[ch]
            u = to_unicode(dots)
            dots_str = "".join(str(d) for d in sorted(dots))
            lines.append(f"{ch} : {u}   (dots {dots_str})")
        # 보기 좋게 2열
        mid=len(lines)//2 + len(lines)%2
        left_col=lines[:mid]; right_col=lines[mid:]
        for i in range(mid):
            l = left_col[i] if i<len(left_col) else ""
            r = right_col[i] if i<len(right_col) else ""
            txt.insert("end", f"{l:<26}   {r}\n")
        txt.configure(state="disabled")

    def clear_result_console(self):
        self.result_txt.configure(state="normal")
        self.result_txt.delete("1.0","end")
        self.result_txt.configure(state="disabled")

    def append_result_console(self, s:str):
        self.result_txt.configure(state="normal")
        self.result_txt.insert("end", s+"\n")
        self.result_txt.see("end")
        self.result_txt.configure(state="disabled")


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root=tk.Tk()
    app=BrailleGameApp(root)
    root.mainloop()

if __name__=="__main__":
    main()
