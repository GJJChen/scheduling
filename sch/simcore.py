import random
from functools import cmp_to_key
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import threading
import time
import multiprocessing
from multiprocessing import Manager
import cProfile
import heapq
import math
class Task:
    def __init__(self, event, timestamp):
        self.event = event
        self.timestamp = timestamp
    def __lt__(self, other):
        # 定义小于号的比较方式，用于 heapq 排序
        return self.timestamp < other.timestamp
    def __repr__(self):
        return f"Task(name={self.name}, priority={self.priority})"
def compare(a, b):
    if a["rate"] < b["rate"]:
        return 1
    return -1
def cmp(a, b):
    if a["timestamp"] > b["timestamp"]:
        return 1
    elif a["timestamp"] == b["timestamp"]:
        if a["size"] < b["size"]:
            return 1
        else:
            return -1
    else:
       return -1
def create_txq_table(user_num, queue_size):
    txq_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tid_table = []
        for tid in range(3):
            #这里使用环形队列实现
            tid_table.append([0 for _ in range(queue_size)])
        txq_table.append(tid_table)
    return txq_table
def create_tx_info_table(user_num):
    tx_info_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tx_info_tid_table = []
        for tid in range(3):
            tx_info_tid_table.append({"tot_size" : 0, "read_ptr" : 0, "write_ptr" : 0})
        tx_info_table.append(tx_info_tid_table)
    return tx_info_table
   
def create_rate_table(user_num):
    rate_table = []
    max_band_width = 2442
    ofo_list = []
    for user_id in range(user_num):
        tid_rate_table = []
        for tid in range(3):
            if tid == 0:
                rand_rate = random.randint(1, 20)
            elif tid == 1:
                rand_rate = random.randint(1, 40)
            elif tid == 2:
                rand_rate = random.randint(1, 100)
            #ratetable包含两个信息，速率和发包间隔
            #tid_rate_table.append(rand_rate * 1000 * 1000)
            if tid == 0:
                rand_interval = random.randint(1, 20)
                #tid_rate_table.append(rand_interval)
            elif tid == 1:
                rand_interval = random.randint(1, 10)
                #tid_rate_table.append(rand_interval)
            else:
                rand_interval = random.randint(1, 3)
                #tid_rate_table.append(rand_interval)
            rate_info = {
                'uid' : user_id,
                'tid' : tid,
                'rate' : rand_rate
            }
            tid_rate_table.append([0, rand_interval])
            ofo_list.append(rate_info)
        #速率先写入0
        rate_table.append(tid_rate_table)
    sorted_list = sorted(ofo_list, key=cmp_to_key(compare))
   
    sum_rate = 0
    for i in range(len(sorted_list)):
        if sum_rate + sorted_list[i]['rate'] < max_band_width:
            uid = sorted_list[i]['uid']
            tid = sorted_list[i]['tid']
            rate_table[uid][tid][0] = sorted_list[i]['rate'] * 1000 * 1000
            #print(f"user {uid}, ac {tid}, set rate is {sorted_list[i]['rate']}")
            sum_rate = sum_rate + sorted_list[i]['rate']
    #调试用
    '''
    for i in range(user_num):
        for j in range(3):
            print(f"user {i}, ac {j}, rate is {rate_table[i][j][0]}")
    '''
    return rate_table
def gen_sched_res(txq_table, tx_info_table, user_num, timestamp_now, queue_size):
    vo_list = []
    vi_list = []
    be_list = []
    #首先从txq_table中取首包做排序
    schedule_res = {"uid" : -1, "ac_type" : -1}
    for uid in range(user_num):
        for ac in range(3):
            w_ptr = tx_info_table[uid][ac]["write_ptr"]
            r_ptr = tx_info_table[uid][ac]["read_ptr"]
            if w_ptr != r_ptr: #队列非空
                pkt_info = txq_table[uid][ac][r_ptr]
                #txq_table[uid][ac].pop(0)
                schedule_info = {
                    "uid" : uid,
                    "ac_type" : ac,
                    "size" : tx_info_table[uid][ac]["tot_size"],
                    "timestamp" : pkt_info['timestamp']
                }
                if ac == 0:
                    vo_list.append(schedule_info)
                elif ac == 1:
                    vi_list.append(schedule_info)
                else:
                    be_list.append(schedule_info)
               
    if len(vo_list) == 0 and len(vi_list) == 0 and len(be_list) == 0:
        schedule_res = {"uid" : -1, "ac_type" : -1}
        return schedule_res
    sorted_vo_list = sorted(vo_list, key=cmp_to_key(cmp))
    sorted_vi_list = sorted(vi_list, key=cmp_to_key(cmp))
    sorted_be_list = sorted(be_list, key=cmp_to_key(cmp))
    #尝试调度VO
    #print(f"timestamp {timestamp_now} vo queue length is {len(sorted_vo_list)}")
    for i in range(len(sorted_vo_list)):
        if timestamp_now - sorted_vo_list[i]["timestamp"] <= 20 and sorted_vo_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vo")
            schedule_res = {"uid" : sorted_vo_list[i]['uid'], "ac_type" : 0}
            return schedule_res
        elif timestamp_now - sorted_vo_list[i]["timestamp"] > 20:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][0]["read_ptr"]
            w_ptr = tx_info_table[uid][0]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][0][r_ptr]['timestamp'] > 20:
                    #记录丢弃长度
                    tx_info_table[uid][0]["tot_size"] = tx_info_table[uid][0]["tot_size"] - txq_table[uid][0][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][0]["read_ptr"] = r_ptr
    #尝试调度VI
    for i in range(len(sorted_vi_list)):
        if timestamp_now - sorted_vi_list[i]["timestamp"] > 20 and timestamp_now - sorted_vi_list[i]["timestamp"] <= 50 and sorted_vi_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vi")
            schedule_res = {"uid" : sorted_vi_list[i]['uid'], "ac_type" : 1}
            return schedule_res
        elif timestamp_now - sorted_vi_list[i]["timestamp"] > 50:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][1]["read_ptr"]
            w_ptr = tx_info_table[uid][1]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][1][r_ptr]['timestamp'] > 50:
                    #记录丢弃长度
                    tx_info_table[uid][1]["tot_size"] = tx_info_table[uid][1]["tot_size"] - txq_table[uid][1][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][1]["read_ptr"] = r_ptr
    #调度BE，假定所有用户的最小需求都是100ms调度一次
    for i in range(len(sorted_be_list)):
        if timestamp_now - sorted_be_list[i]["timestamp"] > 100 and sorted_be_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_be_list[i]['uid'])}, ac type is be")
            schedule_res = {"uid" : sorted_be_list[i]['uid'], "ac_type" : 2}
            return schedule_res
    #没有调度到BE，调度剩余报文最长的用户
    max_pkt_size = 0
    res_uid = -1
    res_ac_type = -1
    #这个时候队列可能已经发生变化
    for i in range(user_num):
        if tx_info_table[i][1]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][1]["tot_size"]
            res_ac_type = 1
    for i in range(user_num):
        if tx_info_table[i][2]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][2]["tot_size"]
            res_ac_type = 2
    #print(f"schedule :: uid is {res_uid}, ac type is {res_ac_type}, pkt_size = {max_pkt_size}")
    #print(f"get_schres:: r_ptr is {tx_info_table[res_uid][res_ac_type]['read_ptr']}, w_ptr is {tx_info_table[res_uid][res_ac_type]['write_ptr']}, remain_size is {tx_info_table[res_uid][res_ac_type]['tot_size']}")
    schedule_res = {"uid" : res_uid, "ac_type" : res_ac_type}
    return schedule_res
   
def schedule(txq_table, tx_info_table, user_num, timestamp, queue_size, event_queue):
    max_rate = 2442 * 1000 * 1000
    #先尝试调度VO
    #all_empty = True
    #print("call gen_sched_res")
    sched_res = gen_sched_res(txq_table, tx_info_table, user_num, timestamp, queue_size)
    '''
    globals_dict = {
        'txq_table': txq_table,
        'tx_info_table': tx_info_table,
        'user_num': user_num,
        'start_time': start_time,
        'queue_size': queue_size,
        'gen_sched_res': gen_sched_res # 确保函数也在子进程中可用
    }
    # 使用 runctx 代替 run，并传入 globals 和locals
    cProfile.runctx(
        'gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)',
        globals_dict,
        {},
        sort='time'
    )
    '''
    #cProfile.run('gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)')
    sched_uid = sched_res["uid"]
    sched_tid = sched_res["ac_type"]
    if sched_uid == -1:
        print(f"nothing to schedule!!")
        heapq.heappush(event_queue, Task("schedule", timestamp + 1))
        return
    print(f"schedule uid is {sched_uid}, tid is {sched_tid}")
    temp_sched_size = 0
    r_ptr = tx_info_table[sched_uid][sched_tid]["read_ptr"]
    w_ptr = tx_info_table[sched_uid][sched_tid]["write_ptr"]
    #print(f"r_ptr is {r_ptr}, w_ptr is {w_ptr}")
    while r_ptr != w_ptr:
        if temp_sched_size == max_rate / 1000 * 4:
            break
        if temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"] <= max_rate / 1000 * 4:
            temp_sched_size = temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"]
            #print(f"r_ptr is {r_ptr}, packet_size is {txq_table[sched_uid][sched_tid][0]['size']}, sched_size is {temp_sched_size}")
            r_ptr = (r_ptr + 1) % queue_size
        else:
            txq_table[sched_uid][sched_tid][r_ptr]["size"] = txq_table[sched_uid][sched_tid][r_ptr]["size"] - max_rate / 1000 * 4 + temp_sched_size
            temp_sched_size = max_rate / 1000 * 4
    tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
   
    tx_info_table[sched_uid][sched_tid]["tot_size"] = tx_info_table[sched_uid][sched_tid]["tot_size"] - temp_sched_size
    send_time = temp_sched_size / max_rate * 1000
    if send_time == 0:
        import pdb;pdb.set_trace()
    #print(f"schedule complete timestamp is {timestamp}, send_time is {send_time}, temp_sched_size is {temp_sched_size}, queue remain size is {tx_info_table[sched_uid][sched_tid]['tot_size']}")
   
    heapq.heappush(event_queue, Task("schedule", timestamp + send_time))
    #print(f"sleep time is {send_time / 1000}")
def traffic_generator(txq_table, user_num, timestamp, rate_table, tx_info_table, queue_size, event_queue):
    for uid in range(user_num):
        for ac_num in range(3):
            if rate_table[uid][ac_num][0] > 0:
                #速率表中指定了发包间隔
                if timestamp % rate_table[uid][ac_num][1] == 0:
                    if (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size == tx_info_table[uid][ac_num]["read_ptr"]:
                        print(f"time {timestamp} uid {uid}, tid {ac_num} queue is full, drop input packet")
                        continue
                    packet_size = np.random.poisson(lam=rate_table[uid][ac_num][0] * 0.001)
                    pkt_info = {}
                    pkt_info['size'] = packet_size
                    pkt_info['timestamp'] = timestamp
                    #存储当前队列缓存包长度
                    #print(f"traffic generate uid : {uid}, ac_type : {ac_num}, pkt_size : {packet_size}, queue_remain len : {tx_info_table[uid][ac_num]['tot_size']}")
                    tx_info_table[uid][ac_num]["tot_size"] = tx_info_table[uid][ac_num]["tot_size"] + packet_size
                    w_ptr = tx_info_table[uid][ac_num]["write_ptr"]
                    txq_table[uid][ac_num][w_ptr] = pkt_info
                    tx_info_table[uid][ac_num]["write_ptr"] = (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size
    heapq.heappush(event_queue, Task("traffic_generate", timestamp + 1))
#def create_stat_table(user_num):
       
if __name__ == "__main__":
    user_num = 32
    simu_time = 10000 #毫秒为单位
    queue_size = 512 #每个tid队列缓存的报文数量
    txq_table = create_txq_table(user_num, queue_size)
    tx_info_table = create_tx_info_table(user_num)
    #schedule_static_table = create_stat_table(user_num)
    #事件队列
    priority_queue = []
    heapq.heappush(priority_queue, Task("traffic_generate", 0))
    heapq.heappush(priority_queue, Task("schedule", 0.5))
    duration_per_time = 10000
    runtimes = 100
    '''
    for i in range(100):
        rate_table = create_rate_table(user_num)
        #import pdb; pdb.set_trace()
        traffic_generate_thread = threading.Thread(target = traffic_generator, args=(txq_table, user_num, simu_time, rate_table, tx_info_table, queue_size))
        schedule_thread = threading.Thread(target=schedule, args=(txq_table, tx_info_table, user_num, simu_time, queue_size))
        traffic_generate_thread.start()
        schedule_thread.start()
        traffic_generate_thread.join()
        schedule_thread.join()
    '''
    rate_table = create_rate_table(user_num)
    while True:
        task = heapq.heappop(priority_queue)
        #print(f"timestamp is {task.timestamp}, event is {task.event}")
        if task.event == "traffic_generate":
            traffic_generator(txq_table, user_num, task.timestamp, rate_table, tx_info_table, queue_size, priority_queue)
        else:
            schedule(txq_table, tx_info_table, user_num, task.timestamp, queue_size, priority_queue)
       
        if math.floor(task.timestamp) % duration_per_time == 0:
            rate_table = create_rate_table(user_num)
        if task.timestamp > runtimes * duration_per_time:
            break