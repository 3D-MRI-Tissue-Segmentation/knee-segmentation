from flask import Flask, escape, request, render_template
from PIL import Image, ImageDraw

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
    
@app.route('/render', methods=["POST"])
def render():
    Femur_Width = int(request.form.get("Femur Width"))
    Tibia_Width = int(request.form.get("Tibia Width"))
    Patella_Width = int(request.form.get("Patella Width"))
    Joint_Space = int(request.form.get("Joint Space"))
    
    FemurPixelWidth = (Femur_Width/100)*75+50
    TibiaPixelWidth = (Tibia_Width/100)*75+50
    PatellaPixelWidth = (Patella_Width/100)+30+10
    JointPixelSpace = (Joint_Space/100)*75
    imgDim = 512
    legCenter = 320
    subpatellarSpace = 30
    
    img = Image.new("RGB", (imgDim, imgDim), color="Black")
    draw = ImageDraw.Draw(img)
    
    #draw femur
    draw.rectangle([(legCenter - FemurPixelWidth, 0), (legCenter + FemurPixelWidth, 0.5*imgDim - JointPixelSpace - FemurPixelWidth)], fill="white", outline=None)
    
    draw.chord([(legCenter - FemurPixelWidth, 0.5*imgDim - JointPixelSpace - 2*FemurPixelWidth), (legCenter + FemurPixelWidth, 0.5*imgDim - JointPixelSpace)], 0, 180, fill="white")
    
    #draw tibia
    draw.rectangle([(legCenter - TibiaPixelWidth, 0.5*imgDim + JointPixelSpace + TibiaPixelWidth), (legCenter + TibiaPixelWidth, imgDim)], fill="white", outline=None)
    
    draw.chord([(legCenter - TibiaPixelWidth, 0.5*imgDim + JointPixelSpace), (legCenter + TibiaPixelWidth, 0.5*imgDim + JointPixelSpace + 2*TibiaPixelWidth)], 180, 360, fill="white")
    
    #draw kneecap
    draw.ellipse([(legCenter - subpatellarSpace - FemurPixelWidth - PatellaPixelWidth, 0.5*imgDim - JointPixelSpace - 2*FemurPixelWidth), (legCenter - subpatellarSpace - FemurPixelWidth, 0.5*imgDim - JointPixelSpace)], fill="white")
    
    #save image
    img.save("static/albert5.jpg")
    return render_template("render.html", pic = "static/albert5.jpg", Femur_Value = Femur_Width, Tibia_Value = Tibia_Width, Patella_Value = Patella_Width, Joint_Value = Joint_Space)


