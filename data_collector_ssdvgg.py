from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
import csv
import time


loop_num = 30

# print(torch.cuda.current_device())


def cal_time():
    torch.cuda.synchronize()
    t = time.time()
    return t


def ssd_vgg(batch_size):
    output_f = open('/home/jian/workspace_cx/pytorch_test/batch_test.txt', 'w')
    output_f.close()
    
    class_names = ['gun','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    model_path = 'models/vgg16-ssd-mp-0_7726.pth'
    image_path = 'gun.jpg'
    
    net = create_vgg_ssd(len(class_names), is_test=True)
    # print(net)
    net.load(model_path)
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    
    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    batch_idx  = 0
    total_time = 0.0
    for i in range(loop_num):
        output_f = open('/home/jian/workspace_cx/pytorch_test/batch_test.txt', 'a')
        output_f.write('New Loop\n')
        output_f.flush()
        output_f.close()
        
        t0 = cal_time()
        boxes, labels, probs = predictor.predict(image, batch_size, 10, 0.4)
        t1 = cal_time()

        batch_idx += 1
        if batch_idx>5:
            total_time += (t1 - t0)
        output_f = open('/home/jian/workspace_cx/pytorch_test/batch_test.txt', 'a')
        output_f.write('Execution time: ' + str(t1 - t0) + '\n')
        # print('Execution time: ' + str(t1 - t0))
        output_f.flush()
        output_f.close()
        # if batch_idx > 1:
        #     break

    output_f = open('/home/jian/workspace_cx/pytorch_test/batch_test.txt', 'a')
    output_f.write('Avg time: ' + str(total_time/25.0) + '\n')
    # print('Avg time: ' + str(total_time/25.0))
    output_f.flush()
    output_f.close()

def read_batch_test_data(running_time_list):

    input_f = open('/home/jian/workspace_cx/pytorch_test/batch_test.txt', 'r')
    data_list = list()

    for line in input_f.readlines():
        line = line.rstrip('\n')
        if 'New Loop' in line:
            each_loop_data_list = list()
        elif 'Execution' in line:
            execution_time = line.split(' ')
            each_loop_data_list.append(float(execution_time[-1]))
            data_list.append(each_loop_data_list)
        elif 'Avg' in line:
            continue
        else:
            each_loop_data_list.append(float(line))

    # print(len(data_list))
    res_list = [i for i in data_list[5]]
    for data in data_list[6:]:
        res_list = [x+y for x,y in zip(res_list, data)]

    res_list = [i/25.0 for i in res_list]

    running_time_list.append(res_list)

    input_f.close()

    return running_time_list


if __name__ == "__main__":

    running_time_list = list()

    for i in range(51, 101):

        ssd_vgg(i)
        read_batch_test_data(running_time_list)
        print('Batchsize Finished: ', i)

    write2csv_tmp_list = zip(*running_time_list)
    write2csv_list = list()
    for i in write2csv_tmp_list:
        write2csv_list.append(list(i))

    write_csv_f = open('../data/data_ssd-vgg16_default.csv', 'w')
    writer = csv.writer(write_csv_f)
    writer.writerows(write2csv_list)
    write_csv_f.close()

