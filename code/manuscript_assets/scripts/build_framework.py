"""Restore the original three-panel framework with the audited study design.
Vector PDF and editable SVG; unchanged robot images come from the original drawio.
Requires reportlab. Coordinates use a top-left origin.
"""
from pathlib import Path
from html import escape
import base64
import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
W, H, SCALE = 1000, 450, 7.05 * 72 / 1000
font_dir = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('FrameworkSans', str(font_dir/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('FrameworkSansBold', str(font_dir/'VeraBd.ttf')))
c = canvas.Canvas(str(FIG / 'revised_protocol.pdf'), pagesize=(W*SCALE, H*SCALE), initialFontName='FrameworkSans')
c.scale(SCALE, SCALE)
c.setTitle('Controlled replays, three-policy training, and strict evaluation')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="1000" height="450" fill="white"/>']
BLUE, RED, GREEN, INK = '#35669a', '#bd6035', '#52816b', '#172d42'

def line(x1,y1,x2,y2,color=BLUE,width=1.6):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width)
    c.line(x1,H-y1,x2,H-y2)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')

def arrow(x1,y1,x2,y2,color=BLUE):
    import math
    line(x1,y1,x2,y2,color)
    a=math.atan2(y2-y1,x2-x1)
    pts=[(x2,y2),(x2-8*math.cos(a-.42),y2-8*math.sin(a-.42)),(x2-8*math.cos(a+.42),y2-8*math.sin(a+.42))]
    p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
    for x,y in pts[1:]:p.lineTo(x,H-y)
    p.close();c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0)
    svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="{color}"/>')

def box(x,y,w,h,fill='#edf3fa',edge=BLUE,r=8):
    c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(edge));c.setLineWidth(1.6)
    c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=1)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')

def text(x,y,lines,size=14,bold=False,color=INK,align='center',gap=None):
    if isinstance(lines,str):lines=[lines]
    font = 'FrameworkSansBold' if bold else 'FrameworkSans'
    max_width = {77:135, 195:380, 228:86, 327:108, 542:240, 550:294, 855:244, 425:232}.get(x)
    if max_width:
        widest = max(pdfmetrics.stringWidth(s,font,size) for s in lines)
        if widest > max_width: size *= max_width / widest
    gap=gap or size*1.3
    c.setFillColor(HexColor(color));c.setFont(font,size)
    for i,s in enumerate(lines):
        yy=y+i*gap
        (c.drawCentredString if align=='center' else c.drawString)(x,H-yy,s)
        svg.append(f'<text x="{x}" y="{yy}" text-anchor="{"middle" if align=="center" else "start"}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(s)}</text>')

def picture(name,x,y,w,h):
    p=FIG/'assets'/name
    c.drawImage(str(p),x,H-y-h,w,h)
    uri='data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}"/>')

line(397,10,397,438,'#a9bacb',1)
line(708,10,708,438,'#a9bacb',1)
text(195,30,'(a) Controlled simulator replays',18,True)
text(550,30,'(b) Fixed-budget training',18,True)
text(855,30,'(c) Strict evaluation',18,True)

picture('replay_scene.jpg',10,110,135,79)
text(77,209,['Shared recorded','qpos / qvel'],14,True)
box(10,270,135,38)
for xx in [44,78,111]:line(xx,270,xx,308,'#8ca7c2',1)
text(27,294,'a0',14);text(61,294,'a1',14);text(94,294,'...',14);text(128,294,'aT-1',10)
text(77,332,['Same sequence','of raw actions'],14,True)
text(327,65,['Illustrative','rendered views'],12,color='#5b6b7a')
for yy,label,col in [(95,['N: nominal'],BLUE),(205,['M: mass and','inertia x 4'],RED),(315,['D: delay of','2 raw steps'],RED)]:
    box(180,yy,96,54,fill='#edf3fa' if col==BLUE else '#fbefea',edge=col)
    text(228,yy+22,label,13,True,color=col)
    picture('robot_view.jpg',300,yy-1,55,55)
    arrow(276,yy+27,297,yy+27,col)
line(161,122,161,342);line(145,150,161,150);line(145,289,161,289)
for yy in [122,232,342]:arrow(161,yy,177,yy)
text(195,405,['Paired replays reuse recorded configurations', 'and actions across N, M, and D.'],14)

policies=[(67,'S: State coverage',['192 source records; N only','384 window records'],BLUE,'#edf3fa'),
          (166,'P: Paired physics',['64 sources; each under N / M / D','128 + 128 + 128 window records'],RED,'#fbefea'),
          (265,'U: Unpaired physics',['192 records; one condition each','128 + 128 + 128 window records'],GREEN,'#eef5f0')]
for yy,title,body,col,fill in policies:
    box(413,yy,254,80,fill,col)
    text(425,yy+23,title,16,True,col,align='left')
    text(425,yy+46,body,13,align='left',gap=18)
    line(667,yy+40,688,yy+40,col)
line(688,107,688,387)
arrow(688,387,671,387)
box(413,365,258,63,'#e4edf6')
text(542,388,'Same model and optimizer',15,True)
text(542,411,'10,000 updates; 3 seeds per policy',13)

box(727,68,256,60)
text(855,91,'64 held-out source episodes',15,True)
text(855,113,'Zero training-episode overlap',13)
arrow(855,128,855,148)
for x,label,col in [(727,'N',BLUE),(793,'M',RED),(859,'D',RED),(925,'MD',RED)]:
    box(x,150,58,36,'#ffffff' if label=='MD' else '#edf3fa',col)
    text(x+29,174,label,15,True,col)
text(855,207,'MD: held-out composition of familiar settings',12)
arrow(855,211,855,230)
box(727,233,256,70)
text(855,257,'Latent prediction',15,True)
text(855,279,['Raw / standardized MSE; cosine error'],12)
text(855,295,'Teacher-forced and free-running',12)
arrow(855,303,855,325)
box(727,328,256,70,'#eef5f0',GREEN)
text(855,352,'Physical position evaluation',15,True)
text(855,373,['Common train-only linear decoders','Block / end effector; persistence'],12,gap=17)
text(855,426,'18 forecast steps = 90 raw action steps',12)

c.showPage();c.save()
svg.append('</svg>')
(FIG/'revised_protocol.svg').write_text('\n'.join(svg))
