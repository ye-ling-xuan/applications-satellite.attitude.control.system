"""Captioned scientific demonstration made from independently verified V3 data."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".artifact_deps"))
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont,ImageOps
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from inertia_generalization.reporting import read_csv,LABELS,COLORS,FILES,FIG_DIR
from inertia_generalization.protocol import RESULTS_DIR,ANALYSIS_DIR

OUT=ROOT/"09_结项文书与答辩"/"07_演示视频"
TMP=ROOT/"tmp/artifacts/video_review"
FPS=10
FONT="C:/Windows/Fonts/msyh.ttc"


def canvas(title,caption=""):
    frame=Image.new("RGB",(1280,720),"white");draw=ImageDraw.Draw(frame)
    draw.text((45,22),title,font=ImageFont.truetype(FONT,30),fill="#172533")
    if caption:draw.text((45,675),caption,font=ImageFont.truetype(FONT,19),fill="#475569")
    return frame


def card(title,paragraphs):
    frame=canvas(title);draw=ImageDraw.Draw(frame);y=150
    font=ImageFont.truetype(FONT,27)
    for text in paragraphs:
        while text:
            cut=min(40,len(text))
            if len(text)>40:
                punctuation=max(text.rfind("，",0,40),text.rfind("。",0,40),text.rfind("；",0,40))
                if punctuation>=20:cut=punctuation+1
            draw.text((60,y),text[:cut],font=font,fill="#263746");y+=47;text=text[cut:]
        y+=15
    return frame


def figure_frame(name,title,caption):
    frame=canvas(title,caption)
    with Image.open(FIG_DIR/(name+".png")) as im:
        im=ImageOps.contain(im.convert("RGB"),(1210,580));frame.paste(im,((1280-im.width)//2,80+(580-im.height)//2))
    return frame


def animation_frames(axis,rows):
    traces={}
    for c in ("PID_N","PID_R","SAC_N","SAC_R"):
        row=next(r for r in rows if r["axis"]==axis and r["controller"]==c and r["rho"]==1.4 and r["sample_id"]==0 and r["model_seed"]==("0" if c.startswith("PID") else "11"))
        with np.load(RESULTS_DIR/row["trace_file"],allow_pickle=False) as data:
            traces[c]={k:data[k].copy() for k in ("time","error_deg","rate_deg_s")}
    pltfont={"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False}
    matplotlib.rcParams.update(pltfont)
    fig=Figure(figsize=(12.4,5.6),dpi=100);agg=FigureCanvasAgg(fig);axs=fig.subplots(1,2)
    lines={}
    for ax,field,label,threshold in zip(axs,("error_deg","rate_deg_s"),("真实姿态误差 / °","真实角速度范数 / °/s"),(1,.5)):
        maximum=max(np.max(t[field]) for t in traces.values())
        ax.set(xlim=(0,20),ylim=(1e-5,max(2,maximum*1.3)),yscale="log",xlabel="仿真时间 / s",ylabel=label)
        ax.axhline(threshold,color="#666666",ls="--",lw=1);ax.axvspan(15,20,color="#dcecef",alpha=.5)
        ax.grid(alpha=.2)
        for c in traces:lines[c,field],=ax.plot([],[],color=COLORS[c],label=LABELS[c],lw=2)
    axs[0].legend(loc="lower left",fontsize=10);fig.tight_layout()
    for index in range(22*FPS):
        t=min(20,index/FPS)
        for c,trace in traces.items():
            cut=np.searchsorted(trace["time"],t,side="right")
            for field in ("error_deg","rate_deg_s"):lines[c,field].set_data(trace["time"][:cut],np.maximum(trace[field][:cut],1e-7))
        agg.draw();plot=Image.fromarray(np.asarray(agg.buffer_rgba())[:,:,:3].copy())
        title=("单轴" if axis=="single" else "三轴")+f"冻结策略部署  ρ=1.4  仿真t={t:04.1f}s"
        frame=canvas(title,"固定初态0、SAC种子11；阴影为末5秒保持区间，代表轨迹不替代全体统计")
        frame.paste(plot,(20,90));yield frame


def main():
    verified=json.loads((ANALYSIS_DIR/"独立核验记录.json").read_text(encoding="utf-8"))
    if verified.get("status")!="passed":raise ValueError("Video requires passed independent verification")
    OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    rows=read_csv(RESULTS_DIR/"episode_metrics.csv")
    conclusions=json.loads((ANALYSIS_DIR/"核心结论.json").read_text(encoding="utf-8"))
    path=OUT/"卫星姿态控制结项演示.mp4"
    intro=card("卫星姿态控制研究演示",["最后一轮研究：惯量域随机化与免重整定泛化边界", "12份SAC模型，每份100万步；780场景，10,920正式回合", "强PID对照，同检查点校准消融，全部训练种子保留", "本视频为数据动画与字幕演示，无配音"])
    success=figure_frame(FILES[0],"全体初态与种子：成功率随惯量变化","训练范围0.7–1.3；0.5、0.6、1.4、1.5为保留域外点，阴影表示种子SD")
    boundaries=figure_frame(FILES[1],"经验可用范围与种子一致性","≥27/30成功；共同区间要求全部种子逐点通过，不推断未测惯量")
    ending=card("研究结论与边界",[r["text"] for r in conclusions["findings"]]+[
        "标称SAC控制代价较PID低54.7%（单轴）、28.7%（三轴），响应较慢", 
        "固定策略、回合内静态惯量、无扰动噪声；控制代价不是实测电能",
        "没有验证在线适应、硬件或在轨性能"])
    writer=imageio_ffmpeg.write_frames(str(path),(1280,720),fps=FPS,codec="libx264",quality=7,
        output_params=["-pix_fmt","yuv420p","-movflags","+faststart"],ffmpeg_log_level="error")
    writer.send(None);count=0
    def add(frame):
        nonlocal count
        writer.send(np.asarray(frame,dtype=np.uint8));count+=1
        if count in (1,100,300,500,700,850):frame.save(TMP/f"frame_{count:04}.jpg",quality=93)
    try:
        for frame,seconds in ((intro,8),(success,16)):
            for _ in range(seconds*FPS):add(frame)
        for axis in ("single","three"):
            for frame in animation_frames(axis,rows):add(frame)
        for frame,seconds in ((boundaries,12),(ending,10)):
            for _ in range(seconds*FPS):add(frame)
    finally:writer.close()
    durations=[(0,8,"研究问题与冻结实验规模"),(8,24,"全体成功率与种子波动"),(24,46,"单轴固定代表轨迹"),(46,68,"三轴固定代表轨迹"),(68,80,"经验可用范围"),(80,90,"实测结论和适用限制")]
    def timecode(t):return f"00:{t//60:02}:{t%60:02},000"
    subtitles="\n\n".join(f"{i+1}\n{timecode(a)} --> {timecode(b)}\n{text}" for i,(a,b,text) in enumerate(durations))+"\n"
    (OUT/"演示章节字幕.srt").write_text(subtitles,encoding="utf-8-sig")
    records={"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "fps":FPS,"frames":count,"duration_s":count/FPS,"resolution":[1280,720],"audio":False,"captions_burned_in":True,
             "fixed_representative":{"rho":1.4,"sample_id":0,"seed":11,"axes":["single","three"]},"visual_review":"pending"}
    (OUT/"视频生成清单.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text("# 结项数据演示视频\n\n90秒，1280×720，10fps，MP4/H.264。画面内含中文说明，无配音，可直接播放；SRT提供章节标记。\n\n动画使用V3已核验真实NPZ，固定ρ=1.4、初态0、SAC种子11，展示四个主控制器。全体统计见图01/02；曲线演示不用于挑选最佳模型。视频不模拟真实卫星外观或真实展开过程。\n\n复现命令：`python tools/build_closing_video.py`。读取核验报告passed后才能生成。\n",encoding="utf-8")
    print(json.dumps(records,ensure_ascii=True))


if __name__=="__main__":main()
