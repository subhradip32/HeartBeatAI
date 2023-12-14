from django.shortcuts import render
import joblib
import sklearn
import csv 

def make_prediction(data):
    model = joblib.load("D:\Code\DT_project\Heart_Beat_ai\heart_beat_ai.ml")
    #we have to take the inputs form the users
    newpredict = model.predict([[data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],1]])
    print(newpredict)
    return float(newpredict)

def get_range():
    max = [57.0,1.0,86.0,1.0,0.61,40.0,6.73,39.0,3.0,1.00,2.0,1.0]
    min = [0.25,0.0,46.0, 0.0,0.01,0.0,3.42,5.5,1.0,0.36,1.0,0.0]
    data = []
    for i in range(0,len(max)-1):
        data = data + [[min[i],max[i]]]
    return data



def main_page(request):
    result = None
    if(request.method == "POST"):
        sur = request.POST.get("survival")
        stalive = request.POST.get("still-alive")
        age_ = request.POST.get("age-at-heart-attack")
        pe = request.POST.get("pericardial-effusion")
        fs = request.POST.get("fractional-shortening")
        ep = request.POST.get("epss")
        lvd = request.POST.get("lvdd")
        wall_motion = request.POST.get("wall-motion-score")
        wall_motion_index = request.POST.get("wall-motion-index")
        mult = request.POST.get("mult")

        _data = [float(sur),float(stalive),float(age_),float(pe),float(fs),float(ep),
                 float(lvd),float(wall_motion),float(wall_motion_index),float(mult)]

        # print(len(_data))
        result = make_prediction(data=_data)
    return render(request,"main.html",{
        "predict": result,
        "max_min":get_range()
    })