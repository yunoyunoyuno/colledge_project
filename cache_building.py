import json;
import copy;
import pickle
import random;
import requests;
import time as T;
import numpy as np;
import pandas as pd;
from sympy import *;
from scipy import interpolate;
import geopy.distance as dist;
from sklearn.linear_model import LinearRegression;

PATH = "new_plan_large.xlsx";
API = 'https://maps-api.rtic-thai.info/v0.1/route'
MATAPI = 'https://maps-api.rtic-thai.info/v0.1/distancematrix'
DT_FORMAT = "%H:%M:%S";
MULTIPLIER = 2; LS_ITER = 1500 ;VNS_ITR = 2;TIME_PORTION = 60;

class Q:
    data = [None]; l = 0; front = 0;
    def __inc(self,j): return (j+1)%len(self.data);
    def enQ(self,obj):
        if(self.l == len(self.data)):
            j = self.front;
            m = [None]*len(self.data)*2;
            for i in range(self.l):
                m[i] = self.data[j]; j = self.__inc(j);
            self.data = m; self.front = 0;
        j = (self.front + self.l)%(len(self.data));
        self.data[j] = obj; self.l += 1;
    def deQ(self):
        e = self.data[self.front];self.data[self.front] = None;
        self.front = self.__inc(self.front); self.l -= 1;
        return e;
    def size(self): return self.l;

class Min_heap:
    def __init__(self,data):
        self.data = data; self.sizes = len(data);
        for k in range(len(data)//2-1,-1,-1): self.__fix_down(k)
    def size(self): return self.sizes;
    def __gt(self,i,j): 
        return self.data[i].fitness < self.data[j].fitness;
    def __swap(self,i,j): self.data[i],self.data[j] = self.data[j],self.data[i];
    def __fix_down(self,k):
        while(2*k+1 < self.sizes):
            c = 2*k+1;
            if(c+1 < self.sizes and self.__gt(c+1,c)): c += 1;
            if(self.__gt(c,k)): self.__swap(c,k)
            k = c;
    def __fix_up(self,c):
        while(c > 0):
            p = (c-1)//2;
            if(self.__gt(c,p)): self.__swap(c,p)
            c = p;
    def deQ(self): 
        self.__swap(0,-1); self.sizes -= 1; 
        e = self.data.pop(); self.__fix_down(0); 
        return e;
    def enQ(self,obj): 
        self.data.append(obj); 
        self.__fix_up(self.sizes); 
        self.sizes += 1

class Vehicle:
    def __init__(self,v):
        d = Utils.d2s;
        self.id = v["vehicle_id"];
        self.loc,self.loc_end = (v["lat_st"],v["lng_st"]),(v["lat_en"],v["lng_en"]);
        self.time_range = (d(v["time_st"]),d(v["time_en"]));
        self.volume,self.weight = v["volume"],v["weight"];

class Stop:
    fitness = None;time  = None;vehicle = None;
    def __init__(self,s):
        d = Utils.d2s;
        self.id = s["stop_id"]; self.loc = (s["lat"],s["lng"]);
        self.service_time = s["service_time"];
        self.time_range = (d(s["time_st"]),d(s["time_en"]))
        self.volume = s["volume"]; self.weight = s["weight"]

class Utils:
    @staticmethod
    #[ |o| <= 8] ;[ >=400 des ]
    def fetch_mat(o=[()],d = [()],mode = "distance",t="08:00:00",isTolls = False):
        o = [{"lat" : e[0],"lng" : e[1]} for e in o];
        d = [{"lat" : e[0],"lng" : e[1]} for e in d];
        data = {"origins": o,"destinations": d,"departureTime": f"2019-04-18T{t}.000Z",
                "routeMode": f"{mode}","avoidTolls": isTolls}
        data = json.dumps(data)
        res = requests.post(MATAPI,data);
        return res.json();

    @staticmethod
    def fetch(o=(18.509218,99.002598),d =(18.259722,99.490636),\
          mode = "time",t="08:00",isTolls = False):
        time = f"2019-04-18T{t}.000Z";
        data = { "origin": { "lat": o[0],"lng": o[1]},
            "destination": {"lat": d[0],"lng": d[1]},
            "departureTime": time,
            "routeMode": f"{mode}","avoidTolls": isTolls }
        data = json.dumps(data);
        res =  requests.post(API,data);
        if(res.json()["status"] != 'OK'): return None
        return res.json()["routes"][0]["legs"][0];

    @staticmethod
    def f1(a,b,mode = "time",t="08:00:00"):
        k1,k2 = (18.509218,99.002598),s_latlng[b]
        if(a == -2): k1,k2 = k2,k1;
        elif(a != -1): k1 = s_latlng[a]
        return Utils.fetch((float(k1[0]),float(k1[1])),(float(k2[0]),float(k2[1])),mode,t);

    @staticmethod
    def generate_stops_cache(latlng,mode="time",time="08:00"):
        for i in range(len(latlng)//8+1):
            for j in range(len(latlng)//400+1):
                origin_set = latlng[8*i:8*i + 8];  
                des_set = latlng[400*j:400*j+400];
                res = Utils.fetch_mat(origin_set,des_set,mode,time);
                if(res["status"] != "OK"): print("error on gsc",time,i*400,400*i+400)
                else: msc[8*i:8*i + 8:,400*j:400*j+400:] = res["results"];
        print("Finsihed Generated! Stops at time",time);


    @staticmethod
    def generate_vehicles_cache(latlng,mode="time",time = "08:00:00"):
        vinit = (18.509218,99.002598);
        mvcout = np.zeros(len(latlng),dtype="object");
        mvcin = np.zeros((len(latlng),1),dtype="object");

        for i in range(len(latlng)//400+1):
            r1 = Utils.fetch_mat([vinit],latlng[i*400:400*i+400],mode,time);
            if(r1["status"] != "OK"): print("error on vout",time,i*400,i*400+400)
            else: mvcout[i*400:i*400+400] = r1["results"][0];

        for j in range(len(latlng)//8+1):
            r2 = Utils.fetch_mat(latlng[8*j:8*j+8],[vinit],mode,time);
            if(r2["status"] != "OK"): print("error on vin",time,j*8,j*8+8)
            else: mvcin[8*j:8*j+8:,:1:] = r2["results"];

        print("Finsihed Generated! Vehicles at time",time);
        mvc[0] = mvcin; mvc[1] = mvcout;

    @staticmethod
    def execute_by_Q(F,latlng,mode,time):
        queue = Q();
        for fn in F: queue.enQ(fn);
        while(queue.size() > 0):
            f = queue.deQ(); f(latlng,mode,time);

    @staticmethod
    def get_data_from_cache(f:Vehicle or Stop,t : Vehicle or Stop,mode = "time"):

        lat_a,lng_a = f.loc; lat_b,lng_b = t.loc;
        
        if(isinstance(f,Stop) and isinstance(t,Stop)):
            k1 = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k1];
            k2 = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k2];
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            r1 = H[h]["s_cache"][a][b][f'{mode}'];
            r2 = H[(h+1)%len(H)]["s_cache"][a][b][f'{mode}'];
            return (r1+r2)//2

        elif(isinstance(f,Vehicle) and isinstance(t,Stop)):
            h,m,s = [int(e) for e in Utils.s2d(f.time_range[0]).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][1]
            c2 = H[(h+1)%len(H)]["v_cache"][1]
            k = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k];
            d1,d2 = c1[b][f'{mode}'],c2[b][f'{mode}'];
            return  (d1+d2)//2;

        elif(isinstance(f,Stop) and isinstance(t,Vehicle)): 
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][0];
            c2 = H[(h+1)%len(H)]["v_cache"][0];
            k = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k];
            d1,d2 = c1[a][0][f'{mode}'],c2[a][0][f'{mode}'];
            return  (d1+d2)//2

        else: return Utils.d_btw_approx(f,t)

    @staticmethod
    def get_data_from_cache2(f:Vehicle or Stop,t : Vehicle or Stop,mode = "time"):

        lat_a,lng_a = f.loc; lat_b,lng_b = t.loc;
        
        if(isinstance(f,Stop) and isinstance(t,Stop)):
            k1 = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k1];
            k2 = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k2];
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            return H[h]["s_cache"][a][b][f'{mode}'];

        elif(isinstance(f,Vehicle) and isinstance(t,Stop)):
            h,m,s = [int(e) for e in Utils.s2d(f.time_range[0]).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][1]
            k = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k];
            return c1[b][f'{mode}'];

        elif(isinstance(f,Stop) and isinstance(t,Vehicle)): 
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][0];
            k = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k];
            return  c1[a][0][f'{mode}']

        else: return Utils.d_btw_approx(f,t)

    @staticmethod
    def get_data_from_cache3(f:Vehicle or Stop,t : Vehicle or Stop,mode = "time"):

        lat_a,lng_a = f.loc; lat_b,lng_b = t.loc;
        
        if(isinstance(f,Stop) and isinstance(t,Stop)):
            k1 = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k1];
            k2 = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k2];
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            r1 = H[h]["s_cache"][a][b][f'{mode}'];
            r2 = H[(h+1)%len(H)]["s_cache"][a][b][f'{mode}'];
            return r1 if m <= 30 else r2;

        elif(isinstance(f,Vehicle) and isinstance(t,Stop)):
            h,m,s = [int(e) for e in Utils.s2d(f.time_range[0]).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][1]
            c2 = H[(h+1)%len(H)]["v_cache"][1]
            k = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k];
            d1,d2 = c1[b][f'{mode}'],c2[b][f'{mode}'];
            return  d1 if m <= 30 else d2;

        elif(isinstance(f,Stop) and isinstance(t,Vehicle)): 
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][0];
            c2 = H[(h+1)%len(H)]["v_cache"][0];
            k = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k];
            d1,d2 = c1[a][0][f'{mode}'],c2[a][0][f'{mode}'];
            return  d1 if m <= 30 else d2;

        else: return Utils.d_btw_approx(f,t)

    @staticmethod
    def generate_points_with_time(f,t,mode="distance"):
        lat_a,lng_a = f.loc; lat_b,lng_b = t.loc;
        points,time = [],nan
        
        if(isinstance(f,Stop) and isinstance(t,Stop)):
            k1 = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k1];
            k2 = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k2];
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            time = f"{h}.{m}";
            points = [(i,H[i]["s_cache"][a][b][f'{mode}']) for i in range(len(H))];
            
        elif(isinstance(f,Vehicle) and isinstance(t,Stop)):
            h,m,s = [int(e) for e in Utils.s2d(f.time_range[0]).split(":")];
            time = f"{h}.{m}";
            k = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k];
            for i in range(len(H)): 
                points.append((i,H[i]["v_cache"][1][b][f'{mode}']));

        elif(isinstance(f,Stop) and isinstance(t,Vehicle)): 
            h,m,s = [int(e) for e in Utils.s2d(f.time).split(":")];
            time = f"{h}.{m}";
            k  = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k];
            for i in range(len(H)): 
                points.append((i,H[h]["v_cache"][0][a][0][f'{mode}']));
        
        if(time == nan): print("GG",f,t);
        return (points,time)

    @staticmethod
    def p2(f:Vehicle or Stop,t : Vehicle or Stop,mode = "distance"):
        points,time = Utils.generate_points_with_time(f,t,mode);
        d = np.array(points);
        x,y = d[:,0], d[:,1];
        fn = np.polyfit(x,y,1);
        p = np.poly1d(fn)
        h,m = [int(e) for e in time.split(".")];
        return p(float(f"{h*8 + int(m//7)}.{int(m%7)}"))
    
    @staticmethod
    def p3(f:Vehicle or Stop,t : Vehicle or Stop,mode = "distance"):
        points,time = Utils.generate_points_with_time(f,t,mode);
        xdata = [point[0] for point in points];
        ydata = [point[1] for point in points];
        tck = interpolate.splrep(xdata,ydata)
        h,m = [int(e) for e in time.split(".")];
        res = interpolate.splev(float(f"{h*(TIME_PORTION) + m*TIME_PORTION//60}.00"), tck)
        return res;
    
    @staticmethod
    def p4(f:Vehicle or Stop,t : Vehicle or Stop,mode = "distance"):
        r1 = Utils.d_btw(f,t);
        r2 = Utils.p2(f,t,"distance");
        r3 = Utils.p3(f,t,"distance");
        r = r1 if abs(r2-r1) >= abs(r2-r3) else r3;
        return r;

    @staticmethod
    def time_btw(a:Vehicle or Stop,b : Vehicle or Stop):
        return Utils.get_data_from_cache(a,b,mode = "duration")

    @staticmethod
    def d_btw(f:Vehicle or Stop,t : Vehicle or Stop):
        return Utils.get_data_from_cache(f,t,"distance")

    @staticmethod
    def d_btw2(f:Vehicle or Stop,t : Vehicle or Stop):
        return Utils.get_data_from_cache2(f,t,"distance")

    @staticmethod
    def d_btw3(f:Vehicle or Stop,t : Vehicle or Stop):
        return Utils.get_data_from_cache3(f,t,"distance")

    @staticmethod
    def time_btw_approx(a:Vehicle or Stop,b : Vehicle or Stop):
        return Utils.d_btw_approx(a,b)

    @staticmethod
    def d_btw_approx(a:Vehicle or Stop,b : Vehicle or Stop):
        lat_a,lng_a = a.loc; lat_b,lng_b = b.loc
        return dist.distance((lat_a, lng_a), (lat_b, lng_b)).km*MULTIPLIER ;
    
    @staticmethod 
    def arrive_time(v,s,now = 0):
        t =  Utils.time_btw(v,s);
        return  s.time_range[0] if t + now < s.time_range[0] else t+now;

    @staticmethod
    def d2s(date): 
        h,m,s =[ int(e) for e in date.split(":")];
        return h*3600 + m*60 + s;

    @staticmethod
    def s2d(t:int): 
        n = {0,1,2,3,4,5,6,7,8,9}
        h = t//3600; m = (t%3600)//60; s = (t%3600)%60;
        if(h in n): h = "0" + str(h);
        if(m in n): m = "0" + str(m);
        if(s in n): s = "0" + str(s);
        return f'{h}:{m}:{s}'


    @staticmethod
    def stop_status(v: Vehicle, s : Stop, assigned_route : [{"stop" : Stop, "time" : int}]):
        last_stop,t  = v,v.time_range[0];
        dt = total_weight = total_volume = 0;

        if(len(assigned_route) > 0):
            last_stop = assigned_route[-1]["stop"]; 
            t,dt = last_stop.time,last_stop.service_time;
            total_weight = sum([e["stop"].weight for e in assigned_route]);
            total_volume = sum([e["stop"].volume for e in assigned_route]);
        
        time_in_route = (t + dt) + Utils.time_btw(last_stop,s); s.time = time_in_route;

        stop_status = time_in_route < s.time_range[1] and \
                total_volume + s.volume <= v.volume and \
                total_weight + s.weight <= v.weight and \
                time_in_route + Utils.time_btw(s,v) + s.service_time < v.time_range[1];
        
        if(stop_status == False): s.time = None;
        return (stop_status,time_in_route); #Not include service time of s
    
    @staticmethod
    def update_time(soln : [{"vehicle":Vehicle,"route":\
                            [{"stop":Stop,"time":int}]}],idx : int):

        current_stop ,time  = soln["vehicle"],soln["vehicle"].time_range[0];
        if(idx > 0): c = soln["route"][idx-1]; current_stop,time = c["stop"],c["time"];

        for k in range(idx,len(soln["route"])):
            e = soln["route"][k];
            time = Utils.arrive_time(current_stop,e["stop"],time);
            service_time = e["stop"].service_time;
            e["time"],time,current_stop = time, time + service_time, e["stop"];
    
    @staticmethod 
    def evaluate(soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}] ):
        d = 0;
        for state in soln:
            current_loc = state["vehicle"];
            time = state["vehicle"].time_range[0];
            for ans in state["route"]:
                d += Utils.d_btw(current_loc,ans["stop"]);
                current_loc = ans["stop"];
                time = ans["time"] + current_loc.service_time;
        if(isinstance(current_loc,Vehicle)): return d;
        return d + Utils.d_btw(current_loc,state["vehicle"]);

        
def cw(v : Vehicle,stops : list):
    for s in stops: s.fitness = Utils.arrive_time(v,s,v.time_range[0]); 
    h,assigned_route,invalid_stops,time = Min_heap(stops),[],[],0;
    while(h.size() > 0):
        s = h.deQ();
        if(s.vehicle == None):
            status,time = Utils.stop_status(v,s,assigned_route);
            if(status == True):
                assigned_route.append({"stop" : s, "time" : time}); 
                s.time = time; s.vehicle = v.id; 
                invalid_stops += h.data; time += s.service_time;
                for stp in stops: stp.fitness = Utils.arrive_time(s,stp,time);
                h = Min_heap(invalid_stops); invalid_stops = []; continue;
        invalid_stops.append(s);
    if(len(assigned_route) > 0): time += Utils.time_btw(v,assigned_route[-1]["stop"]);
    return (assigned_route,time,invalid_stops);

def create_solution_state(stops = [Stop],vehicles = [Vehicle]):
    soln = []; k = 0;
    for k in range(len(vehicles)):
        route,time,remianing_stops = cw(vehicles[k],stops);
        soln.append({"vehicle" : vehicles[k], "route" : route});
        if(len(remianing_stops) == 0): break;
        stops = remianing_stops;
    return soln;

def create_neighbours(solution=[{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    
    # neighbourhood made by remove ans in route1 and insert insert in route2.
    
    new_soln = copy.deepcopy(solution);
    #random select 2 route
    r1,r2 = random.randint(0, len(solution)-1),random.randint(0,len(solution)-2)
    if r1 <= r2 : r2 += 1; # equally random possiblity.
    soln_1,soln_2 = new_soln[r1],new_soln[r2]
    route1,route2 = soln_1["route"], soln_2["route"];
    if(len(route1) == 0): return new_soln; 

    #remove random stop in route1
    rm1 = random.randint(0, len(route1)-1);
    rm_ans = route1[rm1]; route1.remove(rm_ans);
    if(len(route1) > 0) : Utils.update_time(soln_1,rm1);

    #insert removed stop into route2 (find best position to insert)
    min_d = 5e8; min_id = 0;
    current_location,current_time = soln_2["vehicle"],soln_2["vehicle"].time_range[0];

    for ans2 in route2:
        s1,s2 = rm_ans["stop"],ans2["stop"];
        d = Utils.d_btw(current_location,s1) 
        current_time += d + s1.service_time
        d +=  Utils.d_btw(s1,s2) 
        if(d < min_d): min_d = d; min_id = route2.index(ans2);
        
    route2.insert(min_id,rm_ans);
    Utils.update_time(soln_2,min_id);
    return new_soln;


def local_search(soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}],i = 0):
    current_soln = soln;
    for _ in range(LS_ITER):
        neighbours = create_neighbours(current_soln);  i += 1;
        post,pre = Utils.evaluate(neighbours),Utils.evaluate(current_soln);
        if(post < pre):
            current_soln = neighbours;
            print(f"New Solution Found: {post} iter: {i}");
    return i,current_soln;
        
def vns(solution = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    i = 0;best_soln = solution;
    for _ in range(VNS_ITR):
        i,best_soln = local_search(best_soln,i);
        #pertube();
    #print(best_soln);
    
def main():
    solution_states = create_solution_state(stops,vehicles)
    vns(solution_states);

def build_cache(time,H):
    exe_fn = [Utils.generate_vehicles_cache,Utils.generate_stops_cache]; 
    Utils.execute_by_Q(exe_fn,s_latlng,mode = "time",time = time);
    H.append({"s_cache" : msc, "v_cache" : mvc});
    

#Read data.
#random.seed(0)
vehicles = pd.read_excel(PATH,sheet_name='vehicles', header=1,usecols="A:L");
stops_data = pd.read_excel(PATH, sheet_name='stops', header=1,usecols="A:J");
s_latlng = stops_data.to_numpy()[:,1:3];
stops = [Stop(s) for s in stops_data.to_dict('records')];
vehicles = [Vehicle(v) for v in vehicles.to_dict('records')];

print("Building a cache ...");
H,idx_stopstb,idx = [],dict(),0
msc = np.zeros((len(s_latlng),len(s_latlng)),dtype="object");
mvc = np.zeros(2,dtype="object");

for i in range(len(s_latlng)):
    lat,lng = s_latlng[i][0],s_latlng[i][1];
    idx_stopstb[f"[{lat} {lng}]"] = i;

start = T.time(); 
for i in range(24):
    for j in range(0,60,60//TIME_PORTION):
        h = f"0{i}" if i < 10 else f"{i}";
        m = f"0{j}" if j < 10 else f"{j}";
        build_cache(f"{h}:{m}:00",H);
        print("Add to table index",idx); idx += 1;
        mvc = np.zeros(2,dtype="object");
        msc = np.zeros((len(s_latlng),len(s_latlng)),dtype="object");
        
t = T.time() - start ; 
print("Congrats! It's done !!! with time",t/60,"minutes");

sfile = open('pg_c2', 'wb')
pickle.dump(H, sfile,2)
sfile.close()
