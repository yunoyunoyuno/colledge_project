import json;
import copy;
import random;
import pickle;
import requests;
import time as T;
import numpy as np;
import pandas as pd;
from sympy import *;
import geopy.distance as dist;
import matplotlib.pyplot as plt;

PATH = "new_plan_large.xlsx";
API = 'https://maps-api.rtic-thai.info/v0.1/route'
MATAPI = 'https://maps-api.rtic-thai.info/v0.1/distancematrix'
DT_FORMAT = "%H:%M:%S";
MULTIPLIER = 2; LS_ITER = 15 ;VNS_ITR = 2;TIME_PORTION = 60;PTB1=5;PTB2=3;
#MULTIPLIER = 2; LS_ITER = 1600 ;VNS_ITR = 2;TIME_PORTION = 60;PTB1=100;PTB2=150;


class Stack:
    s ,data = 0, np.zeros(1,dtype="object");
    def size(self): return self.s;
    def is_empty(self): return self.s == 0;
    def peek(self): return self.data[self.s-1];
    def push(self,obj):
        if(self.s >= len(self.data)):
            d = np.zeros(len(self.data)*2,dtype="object");
            for i in range(self.s): d[i] = self.data[i];
            self.data = d;
        self.data[self.s] = obj; self.s += 1;
    def pop(self):
        e = self.data[self.s-1]; 
        self.data[self.s-1] = 0.0;self.s -= 1;
        return e;

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

class DBLinkedListWH:
    class LinkedNode:
        def __init__(self,e,p,n):
            self.element,self.prev,self.next = e,p,n;
    s = 0;
    def __init__(self): 
        self.header = self.LinkedNode(None,None,None);
        self.header.next = self.header.prev = self.header;

    def __assert_not_none(self,obj):
        assert(not(obj == None)),"None is not allowed"

    def __assert_in_range(self,i,m):
        assert(not(i < 0 or i > m)),"i must ∈ [0,self.size()-1]"

    def __node_at(self,i):
        c = self.header;
        for k in range(-1,i): c = c.next;
        return c;

    def __node_of(self,obj):
        c = self.header.next;
        while(c != self.header and c.element != obj): c = c.next;
        return c;
    
    def __add_before(self,n,obj):
        new_node = self.LinkedNode(obj,n.prev,n);
        n.prev.next = n.prev = new_node;
        self.s += 1;

    def __remove_node(self,node):
        p = node.prev; n = node.next;
        p.next = n; n.prev = p;
        self.s -= 1;

    def remove(self,e):
        self.__assert_not_none(e);
        n = self.__node_of(e);
        if(n != self.header): self.__remove_node(n);

    def pop(self,i=-1):
        if(i == -1): i = self.s -1;
        self.__assert_in_range(i,self.s-1);
        n = self.__node_at(i);
        self.__remove_node(n);
        return n.element;
    
    def get(self,i):
        self.__assert_in_range(i,self.s-1);
        return self.__node_at(i).element;

    def append(self,obj): self.__add_before(self.header,obj)

    def add(self,i,e):
        self.__assert_in_range(i,self.s+1);
        self.__assert_not_none(e);
        n = self.__node_at(i);
        self.__add_before(n,e);

    def size(self): return self.s;
    
    def toArray(self):
        a = [];
        k = self.header;
        for _ in range(self.s): 
            a.append(k.next.element);
            k = k.next;
        return a;

    def is_empty(self): return self.s == 0;

    def contains(self,obj):
        self.__assert_not_none(obj);
        return self.__node_of(obj) != self.header;

    def set(self,i,obj):
        self.__assert_in_range(i,self.s-1);
        self.__assert_not_none(obj);
        n = self.__node_at(i); n.element = obj;


class Utils:
    
    @staticmethod
    def check_accuracy(time,i):
        print("check accuracy");g = None;
        print(f"from v to every s on {time}");
        v= vehicles[i];
        print("v tart time",Utils.s2d(v.time_range[0]));
        idx = [];fail = [];
        pE = []; dE = []; r = [];ap = [];
        for i in range(100):
            s = stops[i];
            rd = Utils.fetch(v.loc,s.loc,mode="time",t=time)["distance"];
            approx = Utils.d_btw(v,s); r.append(rd); ap.append(approx);
            de = abs(rd-approx); pe = float(de/rd) * 100
            dE.append(de); pE.append(pe);
            if(de <= 100):  idx.append(i);
            else : fail.append(i);
        print("accuracy = ",len(idx),"%");
        print("There are",len(fail)," data which are failed\n");

    #Fetching a matrix distance from actual API
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

    #Fetching the distance between 2 points from original API.
    @staticmethod
    def fetch(o=(18.509218,99.002598),d =(18.259722,99.490636),\
          mode = "time",t="08:00",isTolls = False):
        time = f"2021-08-10T{t}.000Z";
        data = { "origin": { "lat": o[0],"lng": o[1]},
            "destination": {"lat": d[0],"lng": d[1]},
            "departureTime": time,
            "routeMode": f"{mode}","avoidTolls": isTolls }
        data = json.dumps(data);
        res =  requests.post(API,data);
        if(res.json()["status"] != 'OK'): return None
        return res.json()["routes"][0]["legs"][0];

    #getting data from huge cache.
    @staticmethod
    def get_data_from_cache(f:Vehicle or Stop,t : Vehicle or Stop,mode = "time"):

        lat_a,lng_a = f.loc; lat_b,lng_b = t.loc;

        if(isinstance(f,Stop) and isinstance(t,Stop)):
            k1 = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k1];
            k2 = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k2];
            h,m,_ = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            return H[h]["s_cache"][a][b][f'{mode}'];

        elif(isinstance(f,Vehicle) and isinstance(t,Stop)):
            h,m,_= [int(e) for e in Utils.s2d(f.time_range[0]).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][1]
            k = f"[{lat_b} {lng_b}]"; b = idx_stopstb[k];
            return c1[b][f'{mode}'];

        elif(isinstance(f,Stop) and isinstance(t,Vehicle)): 
            h,m,_ = [int(e) for e in Utils.s2d(f.time).split(":")];
            h = h*TIME_PORTION + m*TIME_PORTION//60;
            c1 = H[h]["v_cache"][0];
            k = f"[{lat_a} {lng_a}]"; a = idx_stopstb[k];
            return  c1[a][0][f'{mode}']

        else: return Utils.d_btw_approx(f,t)

    @staticmethod
    def time_btw(a:Vehicle or Stop,b : Vehicle or Stop):
        return Utils.get_data_from_cache(a,b,mode = "duration")

    @staticmethod
    def d_btw(f:Vehicle or Stop,t : Vehicle or Stop):
        return Utils.get_data_from_cache(f,t,"distance")

    @staticmethod
    def d_btw_approx(a:Vehicle or Stop,b : Vehicle or Stop):
        lat_a,lng_a = a.loc; lat_b,lng_b = b.loc
        return dist.distance((lat_a, lng_a), (lat_b, lng_b)).km*MULTIPLIER ;
    
    # using the utmost vehicle end time and assign to Min heap.
    @staticmethod 
    def arrive_time(v,s,now = 0):
        t =  Utils.time_btw(v,s);
        if(t+now > v.time_range[1]):
            return s.time_range[0];
        return  max(s.time_range[0], t + now);

    @staticmethod 
    def real_arrive_time(v,s,now = 0,time = "08:00"):
        t2 = Utils.s2d(time);
        tt =  Utils.fetch(v.loc,s.loc,mode = "time",t = t2)["duration"];
        if(tt+now > v.time_range[1]):
            return s.time_range[0];
        return  max(s.time_range[0], tt + now);

    #convert date to seconds.
    @staticmethod
    def d2s(date): 
        h,m,s =[ int(e) for e in date.split(":")];
        return h*3600 + m*60 + s;
    
    #convert second to date.
    @staticmethod
    def s2d(t:int): 
        n = {0,1,2,3,4,5,6,7,8,9}
        h = t//3600; m = (t%3600)//60; s = (t%3600)%60;
        if(h in n): h = "0" + str(h);
        if(m in n): m = "0" + str(m);
        if(s in n): s = "0" + str(s);
        return f'{h}:{m}:{s}'

    @staticmethod
    def real_updatetime_vns(v : Vehicle,route,idx=0):
        
        current_stop,time = v,0;
        #select the route from the given index  and use that time.
        prev_stop = current_stop;
        
        if(idx > 0):
            c = route[idx-1];idx -= 1;
            current_stop = c["stop"];

        for k in range(idx,len(route)):
            e = route[k]; 
            time = v.time_range[0] + Utils.fetch(v.loc,e["stop"].loc,\
                t=Utils.s2d(v.time_range[0]))["duration"] if k == 0 else \
            prev_stop["time"] + prev_stop["stop"].service_time + \
                Utils.fetch(prev_stop["stop"].loc,e["stop"].loc,\
                    t=Utils.s2d(prev_stop["stop"].time))["duration"];
        
            if(time > 72000): return time;

            e["time"] = time;
            prev_stop = e;

        # return Time to se whether its should insert
        # in this position or take back to the previous state. 
        return time;
    
    #Update time method for VNS procedure.
    @staticmethod
    def updatetime_vns(v : Vehicle,route,idx=0):
        
        current_stop,time = v,0;
        #select the route from the given index  and use that time.
        prev_stop = current_stop;
        
        if(idx > 0):
            c = route[idx-1];idx -= 1;
            current_stop = c["stop"];

        for k in range(idx,len(route)):
            e = route[k]
            time = v.time_range[0] + Utils.time_btw(v,e["stop"]) if k == 0 else \
            prev_stop["time"] + prev_stop["stop"].service_time + \
                Utils.time_btw(prev_stop["stop"],e["stop"]);
        
            if(time > 72000): return time;

            e["time"] = time;
            prev_stop = e;

        # return Time to se whether its should insert
        # in this position or take back to the previous state. 
        return time;

    #status for VNS procedure.
    @staticmethod
    def status_vns(v,s,route,vip = False):
        tt_weight = tt_volume = 0;
        if(len(route) > 0 or vip == True):
            
            #Check for the total weight and total volume condition.
            for i in range(len(route)):
                tt_weight += route[i]["stop"].weight;
                tt_volume += route[i]["stop"].volume;
            check1 = s.weight + tt_weight <= v.weight and \
                s.volume + tt_volume <= v.volume;
            
            #Check for the time condition.
            if(check1):
                for i in range(len(route)):
                    obj = {"stop" : s,"time" : s.time};
                    route.insert(i,obj);
                    time = Utils.updatetime_vns(v,route);
                    route.pop(i);
                    Utils.updatetime_vns(v,route);

                    last = route[-1]
                    finaltime = Utils.time_btw(v,last["stop"]) + \
                        last["time"] + last["stop"].service_time;

                    if(time + finaltime <= v.time_range[1]) : return (True,i);

                return (False,-1);
            
            #return False for invalid weight&volume condition.
        return (False,-1);

    @staticmethod
    def real_status_vns(v,s,route,vip = False):
        tt_weight = tt_volume = 0;
        if(len(route) > 0 or vip == True):
            
            #Check for the total weight and total volume condition.
            for i in range(len(route)):
                tt_weight += route[i]["stop"].weight;
                tt_volume += route[i]["stop"].volume;
            check1 = s.weight + tt_weight <= v.weight and \
                s.volume + tt_volume <= v.volume;
            
            #Check for the time condition.
            if(check1):
                for i in range(len(route)):
                    obj = {"stop" : s,"time" : s.time};
                    route.insert(i,obj);
                    time = Utils.real_updatetime_vns(v,route);
                    route.pop(i);
                    Utils.real_updatetime_vns(v,route);

                    last = route[-1]
                    finaltime = Utils.fetch(v.loc,last["stop"].loc,\
                        t=Utils.s2d(last["stop"].time))["duration"] + \
                        last["time"] + last["stop"].service_time;

                    if(time + finaltime <= v.time_range[1]) : return (True,i);

                return (False,-1);
            
            #return False for invalid weight&volume condition.
        return (False,-1);

    @staticmethod
    def real_stop_status(v: Vehicle, s : Stop, assigned_route : list):
        last_stop,t  = v,v.time_range[0];
        dt = total_weight = total_volume = 0;

        if(len(assigned_route) > 0):
            last_stop = assigned_route[-1]["stop"]; 
            t,dt = last_stop.time,last_stop.service_time;
            total_weight = sum([e["stop"].weight for e in assigned_route]);
            total_volume = sum([e["stop"].volume for e in assigned_route]);
        
        
        time_in_route = (t + dt) + Utils.fetch(last_stop.loc,s.loc,t=Utils.s2d(t))["duration"]; 
        s.time = time_in_route;

        stop_status = time_in_route < s.time_range[1] and \
                total_volume + s.volume <= v.volume and \
                total_weight + s.weight <= v.weight and \
                time_in_route + Utils.fetch(s.loc,v.loc,t=Utils.s2d(s.time))["duration"] + s.service_time < v.time_range[1];
        
        if(stop_status == False): s.time = None;
        return (stop_status,time_in_route); #Not include service time of s
    
    #determine whether we can add more a stop to this route. 
    @staticmethod
    def stop_status(v: Vehicle, s : Stop, assigned_route : list):
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
    
    
    #return total distance from the given solution. ?? why added time ?
    @staticmethod 
    def evaluate(soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}] ):
        d = 0;
        for state in soln:
            current_loc = state["vehicle"];
            time = current_loc.time_range[0];
            for ans in state["route"]:
                d += Utils.d_btw(current_loc,ans["stop"]);
                current_loc = ans["stop"];
                time = ans["time"] + current_loc.service_time;
        #if no route, return maximum number.
        if(isinstance(current_loc,Vehicle)): return 1e10; 
        return d + Utils.d_btw(current_loc,state["vehicle"]);

    def real_evaluate(soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}] ):
        d = 0;
        for state in soln:
            current_loc = state["vehicle"];
            time = current_loc.time_range[0];
            for ans in state["route"]:
                d += Utils.fetch(current_loc.loc,ans["stop"].loc,t=Utils.s2d(time))["distance"];
                current_loc = ans["stop"];
                time = ans["time"] + current_loc.service_time;
        #if no route, return maximum number.
        if(isinstance(current_loc,Vehicle)): return d; 
        return d + Utils.fetch(current_loc.loc,state["vehicle"].loc,t=Utils.s2d(current_loc.time))["distance"];

    # when local search we will remove or add some stop in to the route
    # that why we need to update the time.
    @staticmethod
    def update_time(soln = [{"vehicle":Vehicle,"route": [{"stop":Stop,"time":int}]}],idx = 0):

        current_stop ,time  = soln["vehicle"],soln["vehicle"].time_range[0];
        #select the route from the given index  and use that time.
        if(idx > 0):
            c = soln["route"][idx-1]; 
            current_stop,time = c["stop"],c["time"];
        #for the given 
        for k in range(idx,len(soln["route"])):
            e = soln["route"][k];
            time = Utils.arrive_time(current_stop,e["stop"],time);
            service_time = e["stop"].service_time;
            e["time"],time,current_stop = time, time + service_time, e["stop"];
            
    @staticmethod
    def real_update_time(soln = [{"vehicle":Vehicle,"route": [{"stop":Stop,"time":int}]}],idx = 0):

        current_stop ,time  = soln["vehicle"],soln["vehicle"].time_range[0];
        #select the route from the given index  and use that time.
        if(idx > 0):
            c = soln["route"][idx-1]; 
            current_stop,time = c["stop"],c["time"];
        #for the given 
        for k in range(idx,len(soln["route"])):
            e = soln["route"][k];
            time = Utils.real_arrive_time(current_stop,e["stop"],time,time = time);
            service_time = e["stop"].service_time;
            e["time"],time,current_stop = time, time + service_time, e["stop"];

    
    @staticmethod
    #soln : ["vehicle" : Vehicle, "route", dblinklist];
    def real_update_time_ll(v : Vehicle,dbl_route,idx=0):
        current_stop,time = v,0;
        #select the route from the given index  and use that time.
        prev_stop = current_stop;
        if(idx > 0):
            c = dbl_route.get(idx-1); 
            current_stop = c["stop"];
            #for the given 
        for k in range(idx,dbl_route.size()):
            
            e = dbl_route.get(k);
            time = v.time_range[0] + Utils.fetch(v.loc,e["stop"].loc,\
                t=Utils.s2d(v.time_range[0]))["duration"] if k == 0 else \
            prev_stop["time"] + prev_stop["stop"].service_time + \
                Utils.fetch(prev_stop["stop"].loc,e["stop"].loc,\
                    t=Utils.s2d(t=prev_stop["stop"].time))["duration"];
               
            if(time > 72000): return time;

            e["time"] = time;
            prev_stop = e;

        # return Time to se whether its should insert
        # in this position or take back to the previous state. 
        return time;
    
    @staticmethod
    #soln : ["vehicle" : Vehicle, "route", dblinklist];
    def update_time_ll(v : Vehicle,dbl_route,idx=0):
        current_stop,time = v,0;
        #select the route from the given index  and use that time.
        prev_stop = current_stop;
        if(idx > 0):
            c = dbl_route.get(idx-1); 
            current_stop = c["stop"];
            #for the given 
        for k in range(idx,dbl_route.size()):
            
            e = dbl_route.get(k);

            time = v.time_range[0] + Utils.time_btw(v,e["stop"]) if k == 0 else \
            prev_stop["time"] + prev_stop["stop"].service_time + \
                Utils.time_btw(prev_stop["stop"],e["stop"]);
               
            if(time > 72000): return time;

            e["time"] = time;
            prev_stop = e;

        # return Time to se whether its should insert
        # in this position or take back to the previous state. 
        return time;
    
    @staticmethod
    def status_llroute(v,s,llroute,vip = False):
        tt_weight = tt_volume = 0;
        if(llroute.size() > 0 or vip == True):
            
            #Check for the total weight and total volume condition.
            for i in range(llroute.size()):
                tt_weight += llroute.get(i)["stop"].weight;
                tt_volume += llroute.get(i)["stop"].volume;
            check1 = s.weight + tt_weight <= v.weight and \
                s.volume + tt_volume <= v.volume;
            
            #Check for the time condition.
            if(check1):
                for i in range(llroute.size()):
                    obj = {"stop" : s,"time" : s.time};
                    llroute.add(i,obj);
                    time = Utils.update_time_ll(v,llroute);
                    llroute.pop(i);
                    Utils.update_time_ll(v,llroute);
                    last = llroute.get(llroute.size()-1);
                    finaltime = Utils.time_btw(v,last["stop"]) + \
                        last["time"] + last["stop"].service_time;
                    if(time+finaltime <= v.time_range[1]) : return (True,i);
                return (False,-1);
            
            #return False for invalid weight&volume condition.
        return (False,-1);

    @staticmethod
    def real_status_llroute(v,s,llroute,vip = False):
        tt_weight = tt_volume = 0;
        if(llroute.size() > 0 or vip == True):
            
            #Check for the total weight and total volume condition.
            for i in range(llroute.size()):
                tt_weight += llroute.get(i)["stop"].weight;
                tt_volume += llroute.get(i)["stop"].volume;
            check1 = s.weight + tt_weight <= v.weight and \
                s.volume + tt_volume <= v.volume;
            
            #Check for the time condition.
            if(check1):
                for i in range(llroute.size()):
                    obj = {"stop" : s,"time" : s.time};
                    llroute.add(i,obj);
                    time = Utils.real_update_time_ll(v,llroute);
                    llroute.pop(i);
                    Utils.real_update_time_ll(v,llroute);
                    last = llroute.get(llroute.size()-1);
                    finaltime = Utils.fetch(v.loc,last["stop"].loc,\
                        t=Utils.s2d(last["stop"].time))["duration"] + \
                        last["time"] + last["stop"].service_time;
                    if(time+finaltime <= v.time_range[1]) : return (True,i);
                return (False,-1);
            
            #return False for invalid weight&volume condition.
        return (False,-1);

    @staticmethod
    def clear_stack(s,alldbl,soln):
        issuelist = [];
        while(s.size() > 0):
            stop = s.peek()["stop"];
            for i in range(len(alldbl)):
                stus,j = Utils.status_llroute(soln[i]["vehicle"],stop,alldbl[i],True);
                if(stus):
                    alldbl[i].add(j,s.pop());
                    Utils.update_time_ll(soln[i]["vehicle"],alldbl[i]); 
                    break;
            else : issuelist.append(s.pop());

        return issuelist;
    
    @staticmethod
    def real_clear_stack(s,alldbl,soln):
        issuelist = [];
        while(s.size() > 0):
            stop = s.peek()["stop"];
            for i in range(len(alldbl)):
                stus,j = Utils.real_status_llroute(soln[i]["vehicle"],stop,alldbl[i],True);
                if(stus):
                    alldbl[i].add(j,s.pop());
                    Utils.real_update_time_ll(soln[i]["vehicle"],alldbl[i]); 
                    break;
            else : issuelist.append(s.pop());

        return issuelist;
    
    @staticmethod
    def perturbe(ans):
        soln = copy.deepcopy(ans[0]);
        a1 = {"route" : []};
        while(len(a1["route"]) == 0):
            r1 = random.randint(0,len(soln)-1)
            a1 = soln[r1];
        alldbl = [];
        for x in soln:
            l = DBLinkedListWH();
            if(len(x["route"])>0):
                for e in x["route"]: l.append(e);
            alldbl.append(l)
            
        dbl1 = alldbl[r1];

        s = Stack()

        for j in range(dbl1.size()-1,-1,-1): s.push(dbl1.pop(j));

        # Stage 1 : if there still available space.
        k = 0; i = -1;
        while(k < len(alldbl) and (not s.is_empty())):
            if( k == r1): 
                k += 1; continue;
            v2 = soln[k]["vehicle"];
            s1 = s.peek()["stop"];
            (stus,i) = Utils.status_llroute(v2,s1,alldbl[k]);
            if(stus and i > 0):
                alldbl[k].add(i,{"stop" : s1, "time" : s1.time});
                #After add please update the time.
                Utils.update_time_ll(v2,alldbl[k]);
                k = 0; continue;
            else: k+= 1; 

        # if not, try to eject others route
        c = 0; 
        while(c < 150 and (not s.is_empty())):
            s1 = s.peek();
            r2 = random.randint(0,len(alldbl)-1);
            while(r2 == r1): 
                r2 = random.randint(0,len(alldbl)-1);
            v2 = soln[r2]["vehicle"];
            route2 = alldbl[r2];
            if(route2.size() == 0): c += 1; continue;
            else :
                for i in range(route2.size()):
                    c += 1;
                    temp_route = copy.deepcopy(route2);
                    temp_route.pop(i);
                    if(temp_route.is_empty()): break;
                    #s3ch = route2.toArray(); s32ch = route2.get(i);
                    stus,_ = Utils.status_llroute(v2,s1["stop"],temp_route);
                    if(stus):
                        s.push(route2.pop(i));
                        route2.add(i,s.pop());
                        Utils.update_time_ll(v2,route2)
                        break;

        # Try to remove 2 stops instead of one, for checking the insertion.
        c = 0; 
        while(c < 100 and (not s.is_empty())):
            s1 = s.peek();
            r2 = random.randint(0,len(alldbl)-1);
            while(r2 == r1): 
                r2 = random.randint(0,len(alldbl)-1);
            v2 = soln[r2]["vehicle"];
            route2 = alldbl[r2];
            if(route2.size() == 0): c += 1; continue;
            else :
                for i in range(route2.size()-2):
                    c += 1;
                    temp_route = copy.deepcopy(route2);
                    temp_route.pop(i);
                    temp_route.pop(i+1);
                    if(temp_route.is_empty()): break;
                    #s3ch = route2.toArray(); s32ch = route2.get(i);
                    stus,_ = Utils.status_llroute(v2,s1["stop"],temp_route);
                    if(stus):
                        s.push(route2.pop(i+1));
                        s.push(route2.pop(i))
                        route2.add(i,s.pop());
                        Utils.update_time_ll(v2,route2)
                        break;

        issuelist = Utils.clear_stack(s,alldbl,soln); 

        while(len(issuelist) > 0):  alldbl[r1].append(issuelist.pop())

        Utils.update_time_ll(soln[r1]["vehicle"],alldbl[r1]);

        for i in range(len(alldbl)): soln[i]["route"] = alldbl[i].toArray();

        ans[1] = copy.deepcopy(soln);
    
    @staticmethod
    def plot(dx,y):
        print("plotting the graph")
        plt.plot(dx,y,color='green', marker='o', linestyle='dashed');
        for x,y in zip(dx,y):
            label = "{:.3f}".format(y/1e6)

            plt.annotate(label,(x,y), 
                        textcoords="offset points",
                        xytext=(0,8),
                        ha='left');

        plt.xlabel("iteration");
        plt.ylabel("total distance of all routes");
        plt.show();

    @staticmethod
    def real_perturbe(ans):
        soln = copy.deepcopy(ans[0]);
        a1 = {"route" : []};
        while(len(a1["route"]) == 0):
            r1 = random.randint(0,len(soln)-1)
            a1 = soln[r1];
        alldbl = [];
        for x in soln:
            l = DBLinkedListWH();
            if(len(x["route"])>0):
                for e in x["route"]: l.append(e);
            alldbl.append(l)
            
        dbl1 = alldbl[r1];

        s = Stack()

        for j in range(dbl1.size()-1,-1,-1): s.push(dbl1.pop(j));

        # Stage 1 : if there still available space.
        k = 0; i = -1;
        while(k < len(alldbl) and (not s.is_empty())):
            if( k == r1): k += 1; continue;
            v2 = soln[k]["vehicle"];
            s1 = s.peek()["stop"];
            (stus,i) = Utils.real_status_llroute(v2,s1,alldbl[k]);
            if(stus and i > 0):
                alldbl[k].add(i,{"stop" : s1, "time" : s1.time});
                #After add please update the time.
                Utils.real_update_time_ll(v2,alldbl[k]);
                k = 0; continue;
            else: k+= 1; 

        # if not, try to eject others route
        c = 0; 
        while(c < PTB1 and (not s.is_empty())):
            s1 = s.peek();
            r2 = random.randint(0,len(alldbl)-1);
            while(r2 == r1): r2 = random.randint(0,len(alldbl)-1);
            v2 = soln[r2]["vehicle"];
            route2 = alldbl[r2];
            if(route2.size() == 0): c += 1; continue;
            else :
                for i in range(route2.size()):
                    c += 1;
                    temp_route = copy.deepcopy(route2);
                    temp_route.pop(i);
                    if(temp_route.is_empty()): break;
                    #s3ch = route2.toArray(); s32ch = route2.get(i);
                    stus,_ = Utils.real_status_llroute(v2,s1["stop"],temp_route);
                    if(stus):
                        s.push(route2.pop(i));
                        route2.add(i,s.pop());
                        Utils.real_update_time_ll(v2,route2)
                        break;

        # Try to remove 2 stops instead of one, for checking the insertion.
        c = 0; 
        while(c < PTB2 and (not s.is_empty())):
            s1 = s.peek();
            r2 = random.randint(0,len(alldbl)-1);
            while(r2 == r1): 
                r2 = random.randint(0,len(alldbl)-1);
            v2 = soln[r2]["vehicle"];
            route2 = alldbl[r2];
            if(route2.size() == 0): c += 1; continue;
            else :
                for i in range(route2.size()-2):
                    c += 1;
                    temp_route = copy.deepcopy(route2);
                    temp_route.pop(i);
                    temp_route.pop(i+1);
                    if(temp_route.is_empty()): break;
                    stus,_ = Utils.real_status_llroute(v2,s1["stop"],temp_route);
                    if(stus):
                        s.push(route2.pop(i+1));
                        s.push(route2.pop(i))
                        route2.add(i,s.pop());
                        Utils.real_update_time_ll(v2,route2)
                        break;

        issuelist = Utils.real_clear_stack(s,alldbl,soln); 

        while(len(issuelist) > 0):  alldbl[r1].append(issuelist.pop())

        Utils.real_update_time_ll(soln[r1]["vehicle"],alldbl[r1]);

        for i in range(len(alldbl)): soln[i]["route"] = alldbl[i].toArray();

        ans[1] = copy.deepcopy(soln);
            

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

def real_cw(v : Vehicle,stops : list):
    for s in stops: 
        s.fitness = Utils.real_arrive_time(v,s,v.time_range[0],time=v.time_range[0]); 
    h,assigned_route,invalid_stops,time = Min_heap(stops),[],[],0;
    while(h.size() > 0):
        s = h.deQ();
        if(s.vehicle == None):
            status,time = Utils.real_stop_status(v,s,assigned_route);
            if(status == True):
                assigned_route.append({"stop" : s, "time" : time}); 
                s.time = time; s.vehicle = v.id; 
                invalid_stops += h.data; time += s.service_time;
                for stp in stops: 
                    stp.fitness = Utils.real_arrive_time(s,stp,time,time=time);
                h = Min_heap(invalid_stops); invalid_stops = []; continue;
        invalid_stops.append(s);
    if(len(assigned_route) > 0): 
        time += Utils.fetch(v.loc,assigned_route[-1]["stop"].loc,\
            t=Utils.s2d(assigned_route[-1]["stop"].time))["duration"];
    return (assigned_route,time,invalid_stops);

#all routes with total time each route into 1 collection from CW algorithm.
def create_solution_state(stops = [Stop],vehicles = [Vehicle]):
    soln = []; k = 0;
    for k in range(len(vehicles)):
        route,time,remianing_stops = cw(vehicles[k],stops);
        soln.append({"vehicle" : vehicles[k], "route" : route});
        if(len(remianing_stops) == 0): break;
        stops = remianing_stops;
    return soln;

def real_create_solution_state(stops = [Stop],vehicles = [Vehicle]):
    soln = []; k = 0;
    for k in range(len(vehicles)):
        route,time,remianing_stops = real_cw(vehicles[k],stops);
        soln.append({"vehicle" : vehicles[k], "route" : route});
        if(len(remianing_stops) == 0): break;
        stops = remianing_stops;
    return soln;

def create_neighbours(solution=[{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    
    # neighbourhood made by remove ans in route1 and insert insert in route2.
    dcp = copy.deepcopy;
    new_soln = copy.deepcopy(solution);
    #random select 2 route
    r1,r2 = random.randint(0, len(solution)-1),random.randint(0,len(solution)-2)
    if r1 <= r2 : r2 += 1; # equally random possiblity.
    soln_1,soln_2 = dcp(new_soln[r1]),dcp(new_soln[r2]);
    route1,route2 = soln_1["route"], soln_2["route"];
    if(len(route1) == 0): return new_soln; 

    #remove random stop in route1
    rm1 = random.randint(0, len(route1)-1);
    rm_ans = route1[rm1]; route1.remove(rm_ans);
    if(len(route1) > 0) : Utils.update_time(soln_1,rm1);

    can_added,i = Utils.status_vns(soln_2["vehicle"],rm_ans["stop"],soln_2["route"]);
    if(can_added == False):
        return new_soln;    

    route2.insert(i,rm_ans);
    Utils.updatetime_vns(soln_2["vehicle"],soln_2["route"]);
    new_soln[r1] = soln_1;
    new_soln[r2] = soln_2;
    return new_soln;

def real_create_neighbours(solution=[{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    
    # neighbourhood made by remove ans in route1 and insert insert in route2.
    dcp = copy.deepcopy;
    new_soln = copy.deepcopy(solution);
    #random select 2 route
    r1,r2 = random.randint(0, len(solution)-1),random.randint(0,len(solution)-2)
    if r1 <= r2 : r2 += 1; # equally random possiblity.
    soln_1,soln_2 = dcp(new_soln[r1]),dcp(new_soln[r2]);
    route1,route2 = soln_1["route"], soln_2["route"];
    if(len(route1) == 0): return new_soln; 

    #remove random stop in route1
    rm1 = random.randint(0, len(route1)-1);
    rm_ans = route1[rm1]; route1.remove(rm_ans);
    if(len(route1) > 0) : Utils.real_update_time(soln_1,rm1);

    can_added,i = Utils.real_status_vns(soln_2["vehicle"],rm_ans["stop"],soln_2["route"]);
    if(can_added == False):
        return new_soln;    

    route2.insert(i,rm_ans);
    Utils.real_updatetime_vns(soln_2["vehicle"],soln_2["route"]);
    new_soln[r1] = soln_1;
    new_soln[r2] = soln_2;
    return new_soln;


def local_search(dx,y,soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}],i = 0):
    current_soln = soln;
    for _ in range(LS_ITER):
        neighbours = create_neighbours(current_soln);  i += 1;
        post,pre = Utils.evaluate(neighbours),Utils.evaluate(current_soln);
        if(post < pre):
            current_soln = neighbours;
            dx.append(i); y.append(post);
            print(f"New Solution Found: {post} iter: {i}");
    return i,current_soln;

def real_local_search(soln = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}],i = 0):
    current_soln = soln;
    for _ in range(LS_ITER):
        neighbours = real_create_neighbours(current_soln);  i += 1;
        post,pre = Utils.real_evaluate(neighbours),Utils.real_evaluate(current_soln);
        if(post < pre):
            current_soln = neighbours;
            print(f"New Solution Found: {post} iter: {i}");
    return i,current_soln;

#***
ans = [None,None];
def vns(dx,y,solution = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    i = 0;best_soln = solution;S
    for _ in range(VNS_ITR):
        i,ans[0] = local_search(dx,y,best_soln,i);
        Utils.perturbe(ans);
        v1 = Utils.evaluate(ans[0]);
        v2 = Utils.evaluate(ans[1]);
        if(v2 < v1):
            print(f"New Solution Found: {v2} iter: {i}");
            dx.append(i); y.append(v2);
        best_soln = ans[0] if v1 < v2 else ans[1];
    print("Best solution is",best_soln);
    print("with total distance",Utils.evaluate(best_soln));

def real_vns(solution = [{"vehicle":Vehicle,"route":[{"stop":Stop,"time":int}]}]):
    i = 0;best_soln = solution;S
    for _ in range(VNS_ITR):
        i,ans[0] = real_local_search(best_soln,i);
        Utils.real_perturbe(ans);
        v1 = Utils.real_evaluate(ans[0]);
        v2 = Utils.real_evaluate(ans[1]);
        if(v2 < v1): 
            print(f"New Solution Found: {v2} iter: {i}");
        best_soln = ans[0] if v1 < v2 else ans[1];
        
    print("Best solution is",best_soln);
    print("with total distance",Utils.real_evaluate(best_soln));
   
def main():
    dx,y = [],[]
    solution_states = create_solution_state(stops,vehicles);
    vns(dx,y,solution_states);
    Utils.plot(dx,y);


def main2():
    real_solution_states = real_create_solution_state(stops,vehicles);
    real_vns(real_solution_states);
    

# -------------------------------- Read data. -------------------------------- #
#random.seed(0);

vehicles = pd.read_excel(PATH,sheet_name='vehicles', header=1,usecols="A:L");
stops_data = pd.read_excel(PATH, sheet_name='stops', header=1,usecols="A:J");
s_latlng = stops_data.to_numpy()[:,1:3];

s_upb = 4;v_upb = 4;
stops = [Stop(s) for s in stops_data.to_dict('records')][:s_upb];
vehicles = [Vehicle(v) for v in vehicles.to_dict('records')][:v_upb];

# Un comment this if you want to use the cache.
'''
stops = [Stop(s) for s in stops_data.to_dict('records')];
vehicles = [Vehicle(v) for v in vehicles.to_dict('records')];
'''
# ------------------------- Gather and generate Cache ------------------------ #

# Un comment this if you want to use the cache.
'''
print("Building a little cache ...");
H,idx_stopstb = [],dict();

for i in range(len(s_latlng)):
    lat,lng = s_latlng[i][0],s_latlng[i][1];
    idx_stopstb[f"[{lat} {lng}]"] = i;

st = T.time();
try:
    print("Preparing for cache data ...")
    inf = open('pg_c2', 'rb');
    H = pickle.load(inf);
    inf.close();
except:
    print("No such file");
    inf.close();
print("Loading time",T.time() - st,"s! Cache is ready with size",len(H));
'''

# -------------------------- Main algorithm is here -------------------------- #
#main(); # Un comment this if you want to use the cache

main2(); # Without cache
