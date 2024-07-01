#!/usr/bin/env python
# coding: utf-8

# In[1]:
from nlpir import ictclas, tools
import os
import regex as re
import pandas as pd
import numpy as np

###POS标注

def NLPIR_parser(path): #对已经含有词性标注的文本不再做二次标注
    # tools.update_license() #每个月需要更新一次授权文件，现在为了测试速度暂时将这行注释掉了，但最后必须取消注释
    with open(path,'r',encoding='utf-8') as f:
        ori = f.read()
        if re.search(r'([一-龥0-9]+/[a-z]{1,5}[0-9]? )+',ori) == None:
            text = ori.split('\n')
            text_tagged = ''
            for i in text: #按段分词
                tag = ''
                if i != '':
                    for word, pos in ictclas.segment(i, pos_tagged=True):
                        tag += word+'/'+pos+' '
                tag += '\n'
                text_tagged += tag
        else:
            text_tagged = ori
            print('已存在NLPIR词性标注，不再重复标注！')
    return text_tagged

def pos_output(raw_path): #支持对单个txt和一个文件夹下的所有txt进行词性标注，若原文件夹下存在两个都被NLPIR标注过的同名文件 也会将两个文件都保留
    if raw_path.endswith('.txt'):
        text = NLPIR_parser(raw_path)
        save_path ='\\'.join(raw_path.split('\\')[:-1])+r'\NLPIR'
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        if 'NLPIR_' in raw_path:
            new_file_path = save_path+'\\'+raw_path.split('\\')[-1]
        else:
            new_file_path = save_path+'\\'+'NLPIR_'+raw_path.split('\\')[-1]
        with open(new_file_path,'w',encoding='utf-8') as f_t:
            f_t.write(text)
        final_path = new_file_path
    else:
        file_list = os.listdir(raw_path)
        text = list()
        for file in file_list:
            content = NLPIR_parser(raw_path+'\\'+file)
            text.append(content)
        save_path = raw_path+r'\NLPIR'
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        for i in range(len(text)):
            if 'NLPIR_' in file_list[i]:
                new_file_path = save_path+'\\'+file_list[i]
            else:
                new_file_path = save_path+'\\'+'NLPIR_'+file_list[i]
            if os.path.isfile(new_file_path):
                new_file_path = re.sub(r'\.txt',r'(1).txt',new_file_path)
            with open(new_file_path,'w',encoding='utf-8') as f_t:
                f_t.write(text[i])
        final_path = save_path
    return final_path

# pos_output(r'C:\Users\22961\Documents\SISU\学术创新周\中国政府白皮书3.9\中国政府白皮书3.9') #√

# import math
# from scipy.stats import bartlett
# from scipy.spatial.distance import pdist
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pingouin as pg


## 提取语言特征&normalization

#共用特征
AMP = ['绝对','完全','极其','高度/d','非常','完美','彻底','极了',r'得/.{1,3} 很',r'之.{0,5}极',r'之.{0,5}至']
CAUS = ['因为','之所以','所以','由于','既然','因此','因而','以致','致使','从而','以至','以至于','就在于','是故'] #待补充
CONC = ['固然','尽管','纵然','纵使','纵令','即使']
COND = ['不管','不论','除非','倘','无论','要','只有','若','若是','倘若','倘使','万一','要是','假如','假若','假使','如','如果','只要','如若','要不是']
NEG = ['不/d','不必/d','没有/v','没有/d','没/','无','未/d','未必','别/','弗/','毋/','勿/','莫/','否/','非/b','不用','不得/vi','甭','不曾','未曾','未尝','无须','毋庸']
POMD = ['能/v','能够/v','可能/v','可以/v','得以/v']
PRIV = ['认为','相信','坚信','愿','怀疑','知道','估计','发现','猜','意识到','注意到','留意到','记得','听说','忘记','想','预期','确定','假设','计算','检查','结论','推测','考虑','决定','推断','视为','确定','辨别','梦见','确保','期望','幻想','害怕','感觉','体会','感受','预见','收集','猜测','听到','希望','想象','指示','推断','判断','知道','注意','观察','察觉','推测','预设','假装','证明','推理','回忆','反思','记住','记得','看见','察觉','展示','表示','假设','思考','理解'] #V到 V得
PRMD = ['值得/v','配/v','便于/v','易于/v','难于/v','难以/d']
SUAV = ['坚持','主张','建议','乞求','恳求','请求','要求','命令','打算','强调','劝告','劝说','推荐','询问','保证','决定','渴望','期望','决定','责令','确保','坚持','指示','承诺','祈求','提议','规定']

#朱（2015）
CONA = ['虽','虽然','却','但','但是','然而','而','而是','偏偏','只是','不过','致','不料','岂止','可是','虽然','虽说','相反地']

#庞（2020）
DWNT = ['稍','稍稍','稍微','略','略微'] #待补充
HDG = ['很少','只','大约','大概','逾','约','差不多','几乎','或多或少','也许','有些','有点']
PUBV = ['宣布','公布','宣告','介绍','阐明','报告','承认','否认','反对','肯定','谈论','解释','同意','争论','断言','打赌','自夸','证明','声称','评论','抱怨','坦白','吐露','确认','主张','传达','揭示','呼喊','预测','预言','暗示','提到','承诺','抗议','评论','回复','说','声明','提交','建议','发誓','作证','警告','写']
PLACE = ['四处','到处','随处','就地','沿街','随地','随时随地']
NEMD = [' 应/v','应该','应当','该当',' 须/','须得','犯得着','犯不着','理当','一定/d','必须',' 该/v',r' 要/v .*?/v']
AUX = [' 能/','能够','可能','可以','得以','_NEMD','值得',r' 配/v \S?/.?[^n] ','便于','易于','难于','难以','愿意','乐意','情愿',' 肯/',' 愿/','想要',' 要想/',' 敢/',' 敢于/',' 乐于/']
abstract_noun = ['办法','部分','成绩','错误','道理','地方','法语','方法','方面',
                 '方向','关系','汉语','好处','化学','基础','机会','技术','计划',
                 '健康','教育','精神','经济','经验','科学','历史','民族','内容',
                 '情况','日语','社会','生活','时间','世界','事情','数学','水平',
                 '思想','态度','条件','文化','文学','文艺','物理','消息','新闻',
                 '艺术','意见','意思','意义','友谊','语法','语言','政治','知识',
                 '主意','阿拉伯语','爱情','榜样','本领','本事','本质','比例',
                 '标准','病菌','材料','产量','成分','成果','成就','程度','传统',
                 '措施','道德','德语','地位','法律','范围','方案','方式','方针',
                 '风景','风力','风俗','概念','感情','感想','高度','个体','根本',
                 '构造','关键','观点','规律','规模','过程','和平','环境','集体',
                 '季节','记忆','纪律','价格','价值','阶段','阶级','结构','结论',
                 '精力','距离','看法','理论','理想','理由','利益','立场','力量',
                 '路线','毛病','矛盾','面貌','民主','命运','目标','目的','能力',
                 '品种','前途','强度','情景','情形','情绪','缺点','任务','生命',
                 '时代','时期','事件','事物','事业','手段','手续','速度','特点',
                 '条约','危机','未来','温度','文明','问题','物质','系统','现实',
                 '现象','想法','效果','心得','心情','信心','形式','形势','形象',
                 '形状','兴趣','性格','性质','学问','业务','医学','意志','因素',
                 '印象','营养','优点','原因','原则','愿望','责任','哲学','真理',
                 '政策','制度','秩序','质量','重点','重量','主观','状况','状态',
                 '资料','资源','自由','作用','奥秘','背景','表情','步骤','才能',
                 '财富','草案','产值','常识','长度','场合','场面','成本','程序',
                 '词汇','挫折','代价','地势','地质','法子','分量','风格','风气',
                 '感受','纲领','高潮','高峰','隔阂','个性','根源','功劳','功能',
                 '官僚主义','观念','规则','含量','行业','后果','货币','基层',
                 '机构','疾病','技能','技巧','价钱','见解','阶层','景色','局面',
                 '科技','来源','类型','利润','灵魂','领域','轮廓','逻辑',
                 '马克思主义','毛泽东思想','能量','品德','品质','奇迹','起源',
                 '气氛','气概','区域','权利','权力','热量','色彩','社会主义',
                 '身份','深度','神情','时机','时节','实质','事迹','势力','思维',
                 '条例','途径','物资','闲话','线路','心理','心思','心意','信念',
                 '信息','形态','行为','性能','学科','学说','压力','眼光','意识',
                 '义务','宇宙','语文','元素','原理','运气','战略','整体','正义',
                 '指标','智慧','主权','资本','资本主义','资格','姿势','姿态',
                 '宗教','罪恶','本能','本性','弊病','秉性','病虫害','病毒','才智',
                 '策略','层次','差距','产业','常规','长处','潮流','成效','诚意',
                 '处境','次序','帝国主义','定理','定律','定义','动态','短处',
                 '对策','恩情','法规','法制','范畴','分寸','风度','风光','风尚',
                 '幅度','纲要','格局','格式','功绩','功效','公有制','共产主义',
                 '共性','构想','故障','惯例','规格','国际主义','含义','汉学',
                 '核心','厚度','环节','激情','迹象','记性','角度','教条','教养',
                 '节奏','精华','境地','境界','局势','利弊','路程','密度','面目',
                 '民意','名誉','模式','难度','浓度','偏见','气魄','气势','前程',
                 '潜力','情感','情理','趋势','权威','权益','人道主义','人格',
                 '人生','人性','荣誉','容量','弱点','设施','深浅','神色','神态',
                 '声势','声誉','生产力','生产率','生机','湿度','实力','式样',
                 '世界观','事例','事态','誓言','收益','手艺','顺序','思潮','思绪',
                 '私有制','素质','所有权','所有制','特色','特性','体制','体质',
                 '条款','条理','威望','威信','唯物主义','唯心主义','现状','限度',
                 '效力','效益','心愿','信誉','性情','修养','旋律','眼力','颜色',
                 '要领','意图','意向','欲望','源泉','远景','争端','正气','知觉',
                 '职能','殖民主义','智力','智能','主流','主义','专长','壮志',
                 '准则','资产','宗旨']
EMPH = ['很/d','极/d','挺/d','怪/d','太/d','非常/d','格外/d','十分/d','分外/d','最/d','顶/d','更/d','更加/d','越发/d','越加/d','相当/d','可/d','真/d','好/d','多/d','多么/d','尤其/d','过/d','过于/d',r'是/vshi .?的/ude1 ./w']
TIME = [' 曾/d',' 曾经/',' 一度/',' 业已/',' 已经/',' 早就/',' 终于/',' 到底/',' 才/',' 就/',' 终归/',' 总归/',' 迟早/',' 早晚/',' 即将/',' 将要/',' 就要/',' 快/d',' 正/d',' 正在/',r'在/p \S{1,5}/v',' 常/d',' 常常/',' 时常/',' 往往/',' 每每/',' 时刻/d'' 老/d',' 总/',' 从来/',' 向来/',' 一向/',' 历来/',' 至今/',' 始终/',' 还/d',' 仍然/',' 依然/',' 永远/',' 一直/',' 刚/d',' 刚刚/',' 立刻/',' 立即/',' 马上/',' 一下/d',' 一下子/d',' 当即/',' 顿时/',' 登时/',' 随后/',' 随即/',' 先/d',' 渐渐/',' 逐渐/',' 连忙/',' 偶尔/',' 忽然/',' 永远/',' 又/',' 再/d [^(就业)|(分配)|(保险)|(贷款)]']
        # if re.search(r'([一-龥0-9]+/[a-z]{1,5}[0-9]?_[A-Z的地得\[])+',ori) == None:

def pang_tagger(path):
    with open (path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        line_updated = list()
        for line in lines:
            for i in EMPH:
                for i_0 in list(set(re.findall(i,line))):
                    line = re.sub(i_0,i_0+'_EMPH',line)
            for i in AMP:
                for i_0 in list(set(re.findall(i,line))):
                    line = re.sub(i_0,i_0+'_AMP',line)
                    line = re.sub(r'_AMP/(.{1,2}) ',r'/\1_AMP ',line)
            for i in TIME:
                if i == r'在/p \S{1,5}/v':
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0.split(' ')[0],i_0.split(' ')[0]+'_TIME',line)
                elif i == ' 再/d [^(就业)|(分配)|(保险)|(贷款)]':
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0.split(' ')[1],i_0.split(' ')[1]+'_TIME',line)
                else:
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0,i_0+'_TIME',line)
                        line = re.sub(r'/_TIME(.{1,2}) ',r'/\1_TIME ',line)
            for i in NEMD:
                if i == r' 要/v .*?/v':
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0.split(' ')[1],i_0.split(' ')[1]+'_NEMD',line)
                else:
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0,i_0+'_NEMD',line)
                        line = re.sub(r'_NEMD/(.{1,2}) ',r'/\1_NEMD ',line)
                        line = re.sub(r'/_NEMD(.{1,2}) ',r'/\1_NEMD ',line)
            for i in AUX:
                if 'NEMD' in i:
                    line = re.sub(r'/([^_]{1,2})[_NEMD]{1,} ',r'/\1_AUX_NEMD ',line)
                else:
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0,i_0+'_AUX',line)
                        line = re.sub(r'_AUX/(.{1,2}) ',r'/\1_AUX ',line)
                        line = re.sub(r'/_AUX(.{1,2}) ',r'/\1_AUX ',line)
            line_1 = line.split()
            for i in range(len(line_1)):
                if '_TIME_TIME' in line_1[i]:
                    line_1[i] = '_'.join(line_1[i].split('_')[:2])
                pos = line_1[i].split('/')[-1].split('_')[0]
                if (pos.endswith('d') and pos != 'wd') or pos.startswith(('u','c','p','e','o','y')):
                    line_1[i] += '_LEXICALWORD'
                elif pos.startswith(('n','s','t','f','v','a','b','z','m','q','r')):
                    line_1[i] += '_CONTENTWORD'
                if pos.startswith('n') or pos.endswith('n'):
                    # if any(keyword in line_1[i].split('/')[0] for keyword in abstract_noun):
                    #     line_1[i] += '_ABSTRACTNOUN'
                    if pos == 'vn' or pos == 'an':
                        line_1[i] += '_NOMZ'
                    else:
                        line_1[i] += '_NN'
                elif pos.startswith('v'):#vn vd 在上面都已经被统计过了
                    if any(keyword in line_1[i].split('/')[0] for keyword in PUBV):
                        line_1[i] += '_PUBV'
                    if any(keyword == line_1[i].split('/')[0] for keyword in SUAV):
                        line_1[i] += '_SUAV'
                    if any(keyword in line_1[i].split('/')[0] for keyword in PRIV):
                        line_1[i] += '_PRIV'
                elif pos.startswith('c'):
                    if any(keyword == line_1[i].split('/')[0] for keyword in COND):
                        line_1[i] += '_COND'
                    if any(keyword == line_1[i].split('/')[0] for keyword in CONC):
                        line_1[i] += '_CONC'
                    if any(keyword == line_1[i].split('/')[0] for keyword in CAUS):
                        line_1[i] += '_CAUS'
                elif pos == 'ww':
                    line_1[i] += '_Q'
                if any(keyword in line_1[i].split('/')[0] for keyword in ['我','咱','俺']):
                    line_1[i] += '_FPP1'
                elif any(keyword in line_1[i].split('/')[0] for keyword in ['你','您']):
                    line_1[i] += '_SPP2'
                elif any(keyword in line_1[i].split('/')[0] for keyword in ['他','她']) and pos.startswith('r'):
                    if '其' not in line_1[i].split('/')[0]:
                        line_1[i] += '_TPP3'
                elif '它' in line_1[i].split('/')[0] and pos.startswith('r'):
                    if '其' not in line_1[i].split('/')[0]:
                        line_1[i] += '_PIT'
                if any(keyword in line_1[i].split('/')[0] for keyword in PLACE):
                    line_1[i] += '_PLACE'
                if any(keyword == line_1[i].split('/')[0] for keyword in DWNT):
                    line_1[i] += '_DWNT'
                if any(keyword == line_1[i].split('/')[0] for keyword in HDG):
                    line_1[i] += '_HDG'
                if any(keyword in line_1[i] for keyword in POMD):
                    line_1[i] += '_POMD'
                if line_1[i].startswith(tuple(NEG)): #待改
                    line_1[i] += '_NEG'
                if any(keyword == line_1[i] for keyword in PRMD):
                    line_1[i] += '_PRMD'
            line_updated.append(line_1)
    line_tagged = ''
    for l in line_updated:
        for i in l:
            line_tagged += i+'\n'
    return line_tagged

# pang_tagger(r'C:\Users\22961\Documents\SISU\学术创新周\朱镕基\NLPIR\2000.txt')

# # 1.1 复制庞（2020）框架
def pang_counter(path): #经过MD标注的文件
    with open (path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        token_count = 0
        token_sum_len = 0
        type_list = {}
        lin_fea = {'Token':0,'NN':0,'PIT':0,'PREP':0,'ADJ':0,'CLF':0,'NUM':0,'AWL':0,'TTR':0,'FPP1':0,
               'SPP2':0,'TPP3':0,'Q':0,'[PRIV]':0,'DWNT':0,'DEMP':0,'CAUS':0,'HDG':0,'POMD':0,'[PUBV]':0,
               'VBD':0,'[PEAS]':0,'[PRESP]':0,'NEG':0,'PRN':0,'CONTENT WORD':0,'AUX':0,'LEXICAL WORD':0,
               '的':0,'地':0,'得':0,'PLACE':0,'RB':0,'[SUAV]':0,'NEMD':0,'PRMD':0,'COND':0,
               'PA':0,'CONJ':0,'NOMZ':0,'EMPH':0,'AMP':0,'TIME':0}
        for line in lines: #一词一段
            lin_fea['EMPH'] += len(re.findall('_EMPH',line)) #待检查
            lin_fea['AMP'] += len(re.findall('_AMP',line))
            lin_fea['TIME'] += len(re.findall('_TIME',line))
            lin_fea['NEMD'] += len(re.findall('_NEMD',line))
            lin_fea['AUX'] += len(re.findall('_AUX',line))
            lin_fea['LEXICAL WORD'] += len(re.findall('_LEXICALWORD',line))
            lin_fea['CONTENT WORD'] += len(re.findall('_CONTENTWORD',line))
            lin_fea['NOMZ'] += len(re.findall('_NOMZ',line))
            lin_fea['NN'] += len(re.findall('_NN',line))
            lin_fea['[PUBV]'] += len(re.findall('_PUBV',line))
            lin_fea['[SUAV]'] += len(re.findall('_SUAV',line))
            lin_fea['[PRIV]'] += len(re.findall('_PRIV',line))
            lin_fea['COND'] += len(re.findall('_COND',line))
            lin_fea['CAUS'] += len(re.findall('_CAUS',line))
            lin_fea['FPP1'] += len(re.findall('_FPP1',line))
            lin_fea['SPP2'] += len(re.findall('_SPP2',line))
            lin_fea['TPP3'] += len(re.findall('_TPP3',line))
            lin_fea['PIT'] += len(re.findall('_PIT',line))
            lin_fea['PLACE'] += len(re.findall('_PLACE',line))
            lin_fea['DWNT'] += len(re.findall('_DWNT',line))
            lin_fea['HDG'] += len(re.findall('_HDG',line))
            lin_fea['POMD'] += len(re.findall('_POMD',line))
            lin_fea['NEG'] += len(re.findall('_NEG',line))
            lin_fea['PRMD'] += len(re.findall('_PRMD',line))
            lin_fea['Q'] += len(re.findall('_Q',line))
            if '_' in line:
                pos = line.split('/')[-1].split('_')[0]
            else:
                pos = line.split('/')[-1]
            token_count += 1
            token_sum_len += len(line.split('/')[0])
            if line.split('_')[0] not in type_list.keys():
                type_list[line.split('_')[0]] = 1
            else:
                type_list[line.split('_')[0]] += 1
            # if pos == 'ww':
            #     lin_fea['Q'] += 1
            if pos.startswith('m'):
                lin_fea['NUM'] += 1
            if 'q' in pos: #数量词是数词和量词的组合
                lin_fea['CLF'] += 1
            if pos.startswith('p'):
                lin_fea['PREP'] += 1
                if pos == 'pbei':
                    lin_fea['PA'] += 1
            elif (pos.endswith('d') and pos != 'wd'):
                lin_fea['RB'] += 1
            elif pos.startswith('a'): #an已经被计入名词而ad也已经被计入副词
                lin_fea['ADJ'] += 1
            elif pos.startswith('r'):
                lin_fea['PRN'] += 1 #确认莱比锡标注法中代词的指称
                if pos.startswith('rz'):
                    lin_fea['DEMP'] += 1
            elif pos.startswith('u'):
                if pos == 'ule':
                    lin_fea['VBD'] += 1
                elif pos == 'uguo':
                    lin_fea['[PEAS]'] += 1
                elif pos == 'uzhe':
                    lin_fea['[PRESP]'] += 1
                elif pos == 'ude1':
                        lin_fea['的'] += 1
                elif pos == 'ude2':
                    lin_fea['地'] += 1
                elif pos == 'ude3':
                    lin_fea['得'] += 1
            elif pos.startswith('c'):
                lin_fea['CONJ'] += 1 #待根据洪鹿平（2008）改
        for k,v in lin_fea.items():
            lin_fea[k] = v / token_count * 1000
        lin_fea['AWL'] = token_sum_len / token_count
        lin_fea['TTR'] = len(type_list.keys()) / token_count
        lin_fea['Token'] = token_count
    return lin_fea

# pang_counter(r'C:\Users\22961\Documents\SISU\学术创新周\朱镕基\NLPIR\2000.txt')

def zhu_tagger(path):
    with open (path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        line_updated = list()
        for line in lines:
            for i in EMPH:
                for i_0 in list(set(re.findall(i,line))):
                    line = re.sub(i_0,i_0+'_EMPH',line)
            for i in AMP:
                for i_0 in list(set(re.findall(i,line))):
                    line = re.sub(i_0,i_0+'_AMP',line)
                    line = re.sub(r'_AMP/(.{1,2}) ',r'/\1_AMP ',line)
            for i in TIME:
                if i == r'在/p \S{1,5}/v':
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0.split(' ')[0],i_0.split(' ')[0]+'_TIME',line)
                elif i == ' 再/d [^(就业)|(分配)|(保险)|(贷款)]':
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0.split(' ')[1],i_0.split(' ')[1]+'_TIME',line)
                else:
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0,i_0+'_TIME',line)
                        line = re.sub(r'/_TIME(.{1,2}) ',r'/\1_TIME ',line)
            if '_TIME_TIME' in line:
                line = '_'.join(line.split('_')[:2])
            for i in NEMD:
                if i == r' 要/v .*?/v':
                    line = re.sub(i.split(' ')[1],i.split(' ')[1]+'_NEMD',line)
                else:
                    for i_0 in list(set(re.findall(i,line))):
                        line = re.sub(i_0,i_0+'_NEMD',line)
                        line = re.sub(r'_NEMD/(.{1,2}) ',r'/\1_NEMD ',line)
                        line = re.sub(r'/_NEMD(.{1,2}) ',r'/\1_NEMD ',line)
            line_1 = line.split()
            for i in range(len(line_1)):
                pos = line_1[i].split('/')[-1].split('_')[0]
                if (pos.startswith('v') and not pos.endswith(('n','d'))):#vn vd 在上面都已经被统计过了
                    if any(keyword in line_1[i].split('/')[0] for keyword in PUBV):
                        line_1[i] += '_PUBV'
                    if any(keyword == line_1[i].split('/')[0] for keyword in SUAV):
                        line_1[i] += '_SUAV'
                    if any(keyword in line_1[i].split('/')[0] for keyword in PRIV):
                        line_1[i] += '_PRIV'
                elif pos.startswith('c'):
                    if any(keyword == line_1[i].split('/')[0] for keyword in COND):
                        line_1[i] += '_COND'
                    elif any(keyword == line_1[i].split('/')[0] for keyword in CONC):
                        line_1[i] += '_CONC'
                    elif any(keyword == line_1[i].split('/')[0] for keyword in CAUS):
                        line_1[i] += '_CAUS'
                    elif any(keyword == line_1[i].split('/')[0] for keyword in CONA):
                        line_1[i] += '_CONA'
                if any(keyword in line_1[i].split('/')[0] for keyword in ['我','咱','俺']):
                    line_1[i] += '_FPP1'
                elif any(keyword in line_1[i].split('/')[0] for keyword in ['你','您']):
                    line_1[i] += '_SPP2'
                elif any(keyword in line_1[i].split('/')[0] for keyword in ['他','她']) and pos.startswith('r'):
                    if '其' not in line_1[i].split('/')[0]:
                        line_1[i] += '_TPP3'
                elif '它' in line_1[i].split('/')[0] and pos.startswith('r'):
                    if '其' not in line_1[i].split('/')[0]:
                        line_1[i] += '_PIT'
                if any(keyword in line_1[i].split('/')[0] for keyword in PLACE):
                    line_1[i] += '_PLACE'
                if any(keyword == line_1[i].split('/')[0] for keyword in DWNT):
                    line_1[i] += '_DWNT'
                if any(keyword == line_1[i].split('/')[0] for keyword in HDG):
                    line_1[i] += '_HDG'
                if any(keyword in line_1[i] for keyword in POMD):
                    line_1[i] += '_POMD'
                if line_1[i].startswith(tuple(NEG)): #待改
                    line_1[i] += '_NEG'
                if any(keyword == line_1[i] for keyword in PRMD):
                    line_1[i] += '_PRMD'
            line_updated.append(line_1)
    line_tagged = ''
    for l in line_updated:
        for i in l:
            line_tagged += i+'\n'
    return line_tagged

# zhu_tagger(r'C:\Users\22961\Documents\SISU\学术创新周\朱镕基\NLPIR\2000.txt')
def MDC_tag_output(path,method): #暂时不能像词性标注那样做到能识别词性标注 停止重复标注
    if path.endswith('.txt'):
        if method == 'pang':
            text = pang_tagger(path)
        elif method == 'zhu':
            text = zhu_tagger(path)
        save_path = '\\'.join(path.split('\\')[:-2])+r'\\MDC_'+method[0].upper()
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        with open(save_path+'\\'+'MDC_'+method+'_'+path.split('\\')[-1],'w',encoding='utf-8') as f_t:
            f_t.write(text)
    else:
        file_list = os.listdir(path)
        text = list()
        for file in file_list:
            if method == 'pang':
                content = pang_tagger(path+'\\'+file)
            elif method == 'zhu':
                content = zhu_tagger(path+'\\'+file)
            text.append(content)
        save_path = '\\'.join(path.split('\\')[:-1])+r'\\MDC_'+method[0].upper()
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        for i in range(len(text)):
            with open(save_path+'\\'+'MDC_'+method+'_'+file_list[i],'w',encoding='utf-8') as f_t:
                f_t.write(text[i])
    return

# MDC_tag_output(r'C:\Users\22961\Documents\SISU\学术创新周\中国政府白皮书3.9\中国政府白皮书3.9\NLPIR','zhu')

# # 1.2 复制朱（2015框架）
def zhu_counter(path):
    with open (path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        token_count = 0
        token_sum_len = 0
        type_list = {}
        lin_fea = {'Token':0,'R':0,'FPP1':0,'SPP2':0,'NEG':0,'RB':0,
                   '[PRIV]':0,'Y':0,'RYV':0,'PRMD':0,'RY':0,'TPP3':0,'有':0,
                   'VERB':0,'T':0,'NN':0,'AWL':0,'TTR':0,'VN':0,'B':0,'UDENG':0,
                   'VD':0,'着':0,'VBD':0,'S':0,'Z':0,'地':0,'VF':0,'O':0,
                   'F':0,'ADJ':0,'把':0,'UYY':0,'得':0,'VX':0,'PRN':0,'MQ':0,
                   'DEMP':0,'的':0,'是':0,'AMP':0,'[SUAV]':0,'RZT':0,'E':0,
                   'RZV':0,'CONC':0,'CONJ':0,'CONA':0,'COND':0,'POMD':0,
                   'CAUS':0,'RT':0,'BYPA':0,'DWNT':0,'EMPH':0,'HDG':0,'NEMD':0,'PREP':0,'TIME':0,'PLACE':0,'[PUBV]':0,'AN':0}
        for line in lines:
            lin_fea['EMPH'] += len(re.findall('_EMPH',line)) #待检查
            lin_fea['AMP'] += len(re.findall('_AMP',line))
            lin_fea['TIME'] += len(re.findall('_TIME',line))
            lin_fea['NEMD'] += len(re.findall('_NEMD',line))
            lin_fea['[PUBV]'] += len(re.findall('_PUBV',line))
            lin_fea['[SUAV]'] += len(re.findall('_SUAV',line))
            lin_fea['[PRIV]'] += len(re.findall('_PRIV',line))
            lin_fea['COND'] += len(re.findall('_COND',line))
            lin_fea['CONC'] += len(re.findall('_CONC',line))
            lin_fea['CONA'] += len(re.findall('_CONA',line))
            lin_fea['CAUS'] += len(re.findall('_CAUS',line))
            lin_fea['FPP1'] += len(re.findall('_FPP1',line))
            lin_fea['SPP2'] += len(re.findall('_SPP2',line))
            lin_fea['TPP3'] += len(re.findall('_TPP3',line))
            lin_fea['PLACE'] += len(re.findall('_PLACE',line))
            lin_fea['DWNT'] += len(re.findall('_DWNT',line))
            lin_fea['HDG'] += len(re.findall('_HDG',line))
            lin_fea['POMD'] += len(re.findall('_POMD',line))
            lin_fea['NEG'] += len(re.findall('_NEG',line))
            lin_fea['PRMD'] += len(re.findall('_PRMD',line))
            word = line.split('_')[0]
            if '_' in line:
                pos = line.split('/')[1].split('_')[0]
            else:
                pos = line.split('/')[-1]
            token_count += 1
            token_sum_len += len(line.split('/')[0])
            if word not in type_list.keys():
                type_list[word] = 1
            else:
                type_list[word] += 1
            if pos == 'mq':
                lin_fea['MQ'] += 1
            elif pos.startswith('p'):
                lin_fea['PREP'] += 1
                if pos == 'pba' or word.split('/')[0] == '将':
                    lin_fea['把'] += 1
                elif pos == 'pbei':
                    lin_fea['BYPA'] += 1
            elif pos == 'y':
                lin_fea['Y'] += 1
            elif pos == 'd':
                lin_fea['RB'] += 1
            elif pos == 'e':
                lin_fea['E'] += 1
            elif pos.startswith('b'):
                lin_fea['B'] += 1
            elif pos.startswith('t'):
                lin_fea['T'] += 1
            elif pos.startswith('z'):
                lin_fea['Z'] += 1
            elif pos.startswith('s'):
                lin_fea['S'] += 1
            elif pos.startswith('o'):
                lin_fea['O'] += 1
            elif pos.startswith('f'):
                lin_fea['F'] += 1
            if pos.startswith('n'):
                lin_fea['NN'] += 1
            elif pos.startswith('v'):
                if pos.endswith('n'):
                    lin_fea['VN'] += 1
                else:
                    if pos.endswith('d'):
                        lin_fea['VD'] += 1
                    elif pos == 'v':
                        lin_fea['VERB'] += 1
                    elif pos.endswith('f'):
                        lin_fea['VF'] += 1
                    elif pos.endswith('x'):
                        lin_fea['VX'] += 1
                    elif pos == 'vshi':
                        lin_fea['是'] += 1
                    elif pos == 'vyou':
                        lin_fea['有'] += 1
            elif pos.startswith('a'):
                if pos.endswith('n'):
                    lin_fea['AN'] += 1
                elif not pos.endswith('d'):
                    lin_fea['ADJ'] += 1
            elif pos.startswith('r'):
                lin_fea['PRN'] += 1 #确认莱比锡标注法中代词的指称
                if pos == 'rr':
                    lin_fea['R'] += 1
                elif pos == 'rz':
                    lin_fea['DEMP'] += 1
                elif pos == 'rzt':
                    lin_fea['RZT'] += 1
                elif pos == 'rzv':
                    lin_fea['RZV'] += 1
                elif pos == 'ry':
                    lin_fea['RY'] += 1
                elif pos == 'ryv':
                    lin_fea['RYV'] += 1
                elif pos == 'r':
                    lin_fea['RT'] += 1
            elif pos.startswith('u'):
                if pos == 'ule':
                    lin_fea['VBD'] += 1
                elif pos == 'udeng':
                    lin_fea['UDENG'] += 1
                elif pos == 'uyy':
                    lin_fea['UYY'] += 1
                elif pos == 'uzhe':
                    lin_fea['着'] += 1
                elif pos == 'ude1':
                    lin_fea['的'] += 1
                elif pos == 'ude2':
                    lin_fea['地'] += 1
                elif pos == 'ude3':
                    lin_fea['得'] += 1
                elif pos == 'c':
                    lin_fea['CONJ'] += 1
        for k,v in lin_fea.items(): #Z-SCORE
            lin_fea[k] = (v / token_count) * 1000
        lin_fea['AWL'] = token_sum_len / token_count
        lin_fea['TTR'] = len(type_list.keys()) / token_count
        lin_fea['Token'] = token_count
    return lin_fea

# # # 2. Z-Score Standardization & 维度分计算

# # # 2.1 复制庞（2020）框架

# # In[6]:

def pang_calculator(path): #无法计算单个文本的维度分，目前只能以整个文件夹的形式放入，计算每个文本相对整个文件夹而言的维度分，path格式be like folder = ['C:\\Users\\22961\\Desktop\\帮打印\\1\\1009八中文译本标注完毕\\1009标注完毕\\']
    mean = {'AWL':1.54,'TTR':0.43,'NN':189.5,'PREP':35.95,'ADJ':8.83,'CLF':18.96,'NUM':37.58,'FPP1':19.4,'SPP2':9.54,'[PRIV]':6.01,'DWNT':0.56,'AMP':6.11,'DEMP':11.33,'CAUS':3.48,'HDG':1.32,'POMD':2.42,
    '[PUBV]':0.22,'TPP3':13.33,'VBD':10.43,'[PEAS]':1.71,'[PRESP]':4.14,'NEG':15.34,
    'PRN':121.11,'LEXICAL WORD':11.6048,'CONTENT WORD':24.8993,'的':53.78,'地':2.27,'得':1.75,'NOMZ':13.305,'RB':81.53,
    '[SUAV]':0.47,'NEMD':0.49,'PRMD':1.3,'COND':3.07,
    'PA':1.77,'CONJ':15.45}
    std = {'AWL':0.16,'TTR':0.08,'NN':61.2,'PREP':11.42,'ADJ':7.49,'CLF':10.08,'NUM':20.18,'FPP1':18.05,'SPP2':12.78,'[PRIV]':5.46,'DWNT':1.60,'AMP':5.44,'DEMP':8.32,'CAUS':3.63,'HDG':2.00,'POMD':2.58,
    '[PUBV]':0.66,'TPP3':11.83,'VBD':6.7,'[PEAS]':1.98,'[PRESP]':5.15,'NEG':9.28,
    'PRN':369.913205,'LEXICAL WORD':22.8142,'CONTENT WORD':83.0628,'的':20.39,'地':3.09,'得':1.97,'NOMZ':20.3557,'RB':28.4,
    '[SUAV]':0.96,'NEMD':1.00,'PRMD':1.61,'COND':3.13,
    'PA':2.42,'CONJ':9.15}

    df_list = []
    
    # save_path = path + r'\Statistics'
    # if os.path.exists(save_path):
    #     print(f'文件夹已存在！')
    # else:
    #     os.mkdir(save_path)
    # path_sta = save_path + r'\\Statistics_' + path.split(r'\\')[-3] + r'.csv'
    # path_z = save_path + r'\\Zscores_'+ path.split(r'\\')[-3] + r'.csv'
    # path_d = save_path + r'\\Dimensions_'+ path.split(r'\\')[-3] + r'.csv'
    df_d = pd.DataFrame()
    df_z_ori = pd.DataFrame()

    df_list_1 = []
    
    if path.endswith('.txt'):
        df_total_1 = pd.DataFrame([pang_counter(path)],index=[path.split('\\')[-1]])
    else:
        files = os.listdir(path)
        for fi in files:
            if fi.endswith('.txt'):
                df_new = pd.DataFrame([pang_counter(path+'\\'+fi)],index=[fi])
                # df_new = df
                df_list_1.append(df_new)
                df_list.append(df_new)
        df_total_1 = pd.concat(df_list_1,sort = False,axis = 0, join = 'inner')
        # df_total_1.to_csv(path_sta,encoding='gbk')

    for i in df_total_1.keys():
        if i in mean.keys():
            df_z_ori[i] = (df_total_1[i] - mean[i])/std[i]
        else: #z-score correction
            df_z_ori[i] = 0
            # df_z[i] = df_total_1[i]
            # df[i] = np.where(df[i] > 5, 5, df[i])
    # df_z.to_csv(path_z,encoding='gbk')
    df_z = df_z_ori.fillna(0).apply(lambda x: round(x, 2)) #Z分数保留两位小数

    df_d['Dimension 1'] = -(df_z['AWL']+df_z['TTR']+df_z['NN']+df_z['PREP']+df_z['ADJ']+df_z['CLF']+df_z['NUM'])+(df_z['FPP1']+df_z['SPP2']+df_z['PIT']+df_z['EMPH']+df_z['Q']+df_z['[PRIV]']+df_z['DWNT']+df_z['AMP']+df_z['DEMP']+df_z['CAUS']+df_z['HDG']+df_z['POMD'])
    df_d['Dimension 2'] = df_z['[PUBV]']+df_z['TPP3']+df_z['VBD']+df_z['[PEAS]']+df_z['[PRESP]']+df_z['NEG']
    # df_d['Dimension 3'] = df_z['PRN']+df_z['LEXICAL WORD']+df_z['AUX']+df_z['CONTENT WORD']+df_z['的']+df_z['地']+df_z['得']+df_z['NOMZ']-(df_z['PLACE']+df_z['RB']+df_z['TIME'])
    df_d['Dimension 3'] = df_z['PRN']+df_z['AUX']+df_z['的']+df_z['地']+df_z['得']+df_z['NOMZ']-(df_z['PLACE']+df_z['RB']+df_z['TIME'])
    df_d['Dimension 4'] = df_z['[SUAV]']+df_z['NEMD']+df_z['PRMD']+df_z['COND']
    df_d['Dimension 5'] = df_z['PA']+df_z['CONJ']
    df_d.loc['total']=[np.array(df_d['Dimension 1']).mean(),np.array(df_d['Dimension 2']).mean(),np.array(df_d['Dimension 3']).mean(),np.array(df_d['Dimension 4']).mean(),np.array(df_d['Dimension 5']).mean()]
    # print(df_d)
    # df_d.to_csv(path_d,encoding='gbk')
    # df_total_s = pd.concat(df_list,sort = False,axis = 0, join = 'inner')
    # df_total_s.to_csv('Statistics_total.csv')
    # df_total_s.to_csv('C:\\Users\\22961\\Desktop\\帮打印\\1\\MAT\\Statistics_total.csv')
    return df_total_1, df_z_ori, df_d


# pang_calculator(r'C:\\Users\\22961\\Desktop\\帮打印\\八中译文（已标注）（多余空格已删除）\\MDC_P\\')


# # # 2.2 复制朱（2015）框架

# # In[27]:


def euc_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的欧式距离
    """
    vec1=np.mat(vec1)
    vec2=np.mat(vec2)
    dist1=float(np.sqrt(np.sum(np.square(vec1-vec2))))
    return dist1

def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def se_euc_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的标准化欧式距离
    """
    Vec = np.vstack([vec1, vec2])
    dist1= pdist(Vec, 'seuclidean')
    return dist1[0]

Register = {'Internet Speech':[3.04,-0.12,0.52,0.18,0.32],
            'Talk Show':[0.79,-0.40,3.27,0.31,0.57],
            'Oral narration of Beijingese':[1.97,-1.50,2.39,14.64,-1.08],
            'Debate':[5.34,-3.47,3.13,-0.10,4.43],
            'Court Trial':[-1.13,-7.42,-1.07,-1.58,-0.82],
            'TV Series':[2.51,5.56,-2.34,1.72,-1.36],
            'Natural Conversation':[8.06,-2.32,5.62,5.69,1.68],
            'Academic Paper':[-6.35,-5.75,-1.72,-3.26,-0.99],
            'News':[-4.52,-3.27,-1.43,-3.02,-0.70],
            'Fiction':[0.44,8.87,-0.85,0.08,-0.75],
            'Biography':[-1.52,1.14,-0.03,-1.14,0.62],
            'Essay':[0.10,6.77,0.78,-0.74,0.93],
            'Magazine':[-5.80,-2.00,-1.72,-2.55,-0.27],
            'Official Document':[-2.66,-7.58,-4.47,-4.33,-3.04],
            'Popular Lore':[-2.57,7.32,-3.24,-1.36,-1.08],
            'Religious Writing':[-3.06,-2.03,0.18,-1.80,1.06]}

mean = {'R':44.32,'FPP1':19.4,'SPP2':9.54,'NEG':15.34,'RB':81.53,'[PRIV]':6.01,'Y':18.20,'RYV':2.46,'PRMD':1.3,
        'RY':4.24,'TPP3':13.33,'有':8.07,'VERB':160.76,'T':12.26,'NN':189.5,'AWL':1.54,'TTR':0.43,'VN':23.96,'B':8.83,'UDENG':1.25,'VD':0.97,
        '着':4.14,'VBD':10.43,'S':4.52,'Z':1.64,'地':2.27,'VF':12.66,'O':0.58,'F':19.48,'ADJ':36.43,'把':1.99,'UYY':0.74,'得':1.75,'VX':1.02,'PRN':121.11,'MQ':5.83,
        'DEMP':11.33,'的':53.78,'是':20.23,'AMP':6.11,'[SUAV]':0.47,'RZT':0.37,'E':2.05,'RZV':11.45,'CONC':0.4,'CONJ':15.45,'CONA':3.88,'COND':3.07,'POMD':2.42,
        'CAUS':3.48,'RT':2.94}
std = {'R':33.34,'FPP1':18.05,'SPP2':12.78,'NEG':9.28,'RB':28.4,'[PRIV]':5.46,'Y':21.04,'RYV':2.65,'PRMD':1.61,
       'RY':4.48,'TPP3':11.83,'有':5.40,'VERB':26.08,'T':9.42,'NN':61.2,'AWL':0.16,'TTR':0.08,'VN':24.28,'B':7.49,'UDENG':2.29,'VD':1.36,
       '着':5.15,'VBD':6.7,'S':3.82,'Z':2.33,'地':3.09,'VF':8.45,'O':1.37,'F':7.94,'ADJ':13.19,'把':2.33,'UYY':1.12,'得':1.97,'VX':1.81,'PRN':369.913205,'MQ':4.73,
       'DEMP':8.32,'的':20.39,'是':12.31,'AMP':5.44,'[SUAV]':0.96,'RZT':1.05,'E':5.98,'RZV':8.55,'CONC':0.76,'CONJ':9.15,'CONA':3.11,'COND':3.13,'POMD':2.58,
       'CAUS':3.63,'RT':2.64}


# # In[30]:


folder = ['C:\\Users\\22961\\Documents\\SISU\\学位论文\\data\\语料\\2000-2003汉\\ST_2000-2003汉\\','C:\\Users\\22961\\Documents\\SISU\\学位论文\\data\\语料\\2004-2013汉\\ST_2004-2013汉\\','C:\\Users\\22961\\Documents\\SISU\\学位论文\\data\\语料\\2014-2023汉\\ST_2014-2023汉\\']

def zhu_calculator(path): #要改成txt文件的绝对路径
    df_list_s = []
    df_z = pd.DataFrame()
    df_z_ori = pd.DataFrame()
    df_d = pd.DataFrame()

    # files = os.listdir(path)

    # save_path = path + r'\Statistics'
    # if os.path.exists(save_path):
    #     print(f'文件夹已存在！')
    # else:
    #     os.mkdir(save_path)

    # path_sta = save_path + r'\\Statistics_' + path.split(r'\\')[-3] + r'.csv'
    # path_z = save_path + r'\\Zscores_'+ path.split(r'\\')[-3] + r'.csv'
    # path_d = save_path + r'\\Dimensions_'+ path.split(r'\\')[-3] + r'.csv'
    
    # files = os.listdir(fo)
    if path.endswith('.txt'):
        df_total_s = pd.DataFrame([zhu_counter(path)],index=[path.split('\\')[-1]])
    else:
        files = os.listdir(path)
        for fi in files:
            if fi.endswith('.txt'):
                df_s = pd.DataFrame([zhu_counter(path+'\\'+fi)],index=[fi])
                df_list_s.append(df_s)
        df_total_s = pd.concat(df_list_s,sort = False,axis = 0, join = 'inner')
        # df_total_s.to_csv(path_sta,encoding='gbk')
    
    for i in mean.keys():
        df_z_ori[i] = (df_total_s[i] - mean[i])/std[i]
    # df_z_ori.to_csv(path_z,encoding='gbk')
    df_z = df_z_ori.fillna(0).apply(lambda x: round(x, 2)) #Z分数保留两位小数

    # df_d['Dimension 1'] = (df_z['R']+df_z['FPP1']+df_z['SPP2']+df_z['NEG']+df_z['RB']+df_z['[PRIV]']+df_z['Y']+df_z['RYV']+df_z['PRMD']+df_z['RY']+df_z['TPP3']+df_z['有']+df_z['VERB']+df_z['VF']+df_z['T'])-(df_z['NOUN']+df_z['AWL']+df_z['TTR']+df_z['VN']+df_z['B']+df_z['UDENG']+df_z['VD']+df_z['VX'])
    df_d['Dimension 1'] = (df_z['R']+df_z['FPP1']+df_z['SPP2']+df_z['NEG']+df_z['RB']+df_z['[PRIV]']+df_z['Y']+df_z['RYV']+df_z['PRMD']+df_z['RY']+df_z['TPP3']+df_z['有']+df_z['VERB']+df_z['T'])-(df_z['NN']+df_z['AWL']+df_z['TTR']+df_z['VN']+df_z['B']+df_z['UDENG']+df_z['VD'])
    # df_d['Dimension 2'] = (df_z['着']+df_z['了']+df_z['S']+df_z['Z']+df_z['地']+df_z['VF']+df_z['O']+df_z['F']+df_z['ADJ']+df_z['把']+df_z['UYY']+df_z['得']+df_z['TTR'])-(df_z['AWL']+df_z['VX']+df_z['DEMP'])
    df_d['Dimension 2'] = (df_z['着']+df_z['VBD']+df_z['S']+df_z['Z']+df_z['地']+df_z['VF']+df_z['O']+df_z['F']+df_z['ADJ']+df_z['把']+df_z['UYY']+df_z['得'])-df_z['VX']
    # df_d['Dimension 3'] = df_z['PRN']+df_z['MQ']+df_z['DEMP']+df_z['的']+df_z['是']+df_z['AMP']+df_z['CAUS']+df_z['[SUAV]']-df_z['CONC']
    df_d['Dimension 3'] = df_z['PRN']+df_z['MQ']+df_z['DEMP']+df_z['的']+df_z['是']+df_z['AMP']+df_z['[SUAV]']
    # df_d['Dimension 4'] = df_z['RZT']+df_z['E']+df_z['RZV']+df_z['RY']+df_z['DEMP']
    df_d['Dimension 4'] = df_z['RZT']+df_z['E']+df_z['RZV']
    # df_d['Dimension 5'] = df_z['CONC']+df_z['CONJ']+df_z['CONA']+df_z['COND']+df_z['POMD']+df_z['CAUS']-(df_z['RT']+df_z['O'])
    df_d['Dimension 5'] = df_z['CONC']+df_z['CONJ']+df_z['CONA']+df_z['COND']+df_z['POMD']+df_z['CAUS']-df_z['RT']
    df_d.loc['total']=[np.array(df_d['Dimension 1']).mean(),np.array(df_d['Dimension 2']).mean(),np.array(df_d['Dimension 3']).mean(),np.array(df_d['Dimension 4']).mean(),np.array(df_d['Dimension 5']).mean()]
    Cloest = []
    for i in range(df_d.shape[0]):
        dis = {}
        row = np.array(df_d.iloc[i]).tolist()
        for r in Register.keys():
            dis[r] = euc_dist(row,Register[r])
        bestmatch = ''
        minValue=min(dis.values())
        for k,v in dis.items():
            if v==minValue:
                bestmatch += f'{k} '
        Cloest.append(bestmatch)
    df_d['Cloest Text Type'] = Cloest
    # df_d.to_csv(path_d,encoding='gbk')
    return df_total_s, df_z_ori, df_d

# zhu_calculator(r'C:\\Users\\22961\\Desktop\\帮打印\\八中译文（已标注）（多余空格已删除）\\MDC_Z\\')

def MDC_result_output(path,method): #暂时不能像词性标注那样做到能识别词性标注 停止重复标注
    if path.endswith('.txt'):
        save_path =  '\\'.join(path.split('\\')[:-1]) + r'\Statistics'
        print(save_path)
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        path_sta = save_path + r'\\Statistics_' + path.split('\\')[-1].rstrip('.txt') + r'.csv'
        print(path_sta)
        path_z = save_path + r'\\Zscores_'+ path.split('\\')[-1].rstrip('.txt') + r'.csv'
        path_d = save_path + r'\\Dimensions_'+ path.split('\\')[-1].rstrip('.txt') + r'.csv'
        if method == 'pang':
            df_total_s,df_z_ori,df_d = pang_calculator(path)
        elif method == 'zhu':
            df_total_s,df_z_ori,df_d = zhu_calculator(path)
        df_total_s.to_csv(path_sta,encoding='gbk')
        df_z_ori.to_csv(path_z,encoding='gbk')
        df_d.to_csv(path_d,encoding='gbk')
    else:
        save_path = r'\\'.join(path.split(r'\\')[:-1]) + r'\Statistics'
        # print(save_path)
        if os.path.exists(save_path):
            print(f'文件夹已存在！')
        else:
            os.mkdir(save_path)
        path_sta = save_path + r'\\Statistics_' + path.split('\\')[-1] + r'.csv'
        path_z = save_path + r'\\Zscores_'+ path.split('\\')[-1] + r'.csv'
        path_d = save_path + r'\\Dimensions_'+ path.split('\\')[-1] + r'.csv'
        if method == 'pang':
            df_total_s,df_z_ori,df_d = pang_calculator(path)
        elif method == 'zhu':
            df_total_s,df_z_ori,df_d = zhu_calculator(path)
        df_total_s.to_csv(path_sta,encoding='gbk')
        df_z_ori.to_csv(path_z,encoding='gbk')
        df_d.to_csv(path_d,encoding='gbk')
        # file_list = os.listdir(path)
        # text = list()
        # for file in file_list:
            
        #     elif method == 'zhu':
        #         content = zhu_calculator(path+'\\'+file)
        #     text.append(content)
        # save_path = '\\'.join(path.split('\\')[:-1])+r'\\MDC_'+method[0].upper()
        # if os.path.exists(save_path):
        #     print(f'文件夹已存在！')
        # else:
        #     os.mkdir(save_path)
        # for i in range(len(text)):
        #     with open(save_path+'\\'+'MDC_'+method+'_'+file_list[i],'w',encoding='utf-8') as f_t:
        #         f_t.write(text[i])
    return

# MDC_result_output(r'C:\Users\22961\Documents\SISU\学术创新周\中国政府白皮书3.9\中国政府白皮书3.9\MDC_Z','zhu')
# MDC_result_output(r'C:\Users\22961\Documents\SISU\学术创新周\NLPIR\MDC_P\MDC_pang_NLPIR_2000.txt','pang')


# turn_on_or_off = input("本脚本支持对UTF-8编码的txt完成多维分析标注和维度分计算^^\n开始使用本工具请输入“0”以外任意字符，结束使用请输入“0”：")
# while turn_on_or_off != "0":
#     command = int(input("使用标注功能请输入“1”，使用维度分计算功能请输入“2”，结束使用请输入“0”："))
#     if command == 1:
#         txt_folder_path = input("请输入需要处理的文件夹或文件的绝对路径（示例格式为：C:\\\\AA\\\\BB 或 C:\\\\AA\\\\BB\\\\CC.txt）：")
#         status = int(input("采用庞双子（2021）分析框架请输入“1”，采用朱晓楠（2015）分析框架请输入“2”，结束使用请输入“0”："))
#         pos_path = pos_output(txt_folder_path)
#         if status == 1:
#             MDC_tag_output(pos_path,'pang')
#         elif status == 2:
#             MDC_tag_output(pos_path,'zhu')
#         else:
#             break
#         # try:
#         #     txt_to_csv(status,txt_folder_path)
#         #     if status == 0:
#         #         combine_or_not = input("需要将每个文件夹所对应的csv合并请输入“1”，否则请按回车键：")
#         #         if combine_or_not == "1":
#         #             target_name = input("为这个csv文件取个名字吧：")
#         #             try:
#         #                 csv_combiner(txt_folder_path,target_name)
#         #             except Exception as e:
#         #                 if "codec can't decode byte" in e:
#         #                     print('本工具目前仅支持UTF-8编码的csv文件合并，请修改相应csv文件编码后重试！')
#         #                 else:
#         #                     print("except",e)
#         # except Exception as e:
#         #     print("except:",e)
#     elif command == 2:
#         txt_folder_path = input("请输入需要处理的文件夹或文件的绝对路径（示例格式为：C:\\\\AA\\\\BB 或 C:\\\\AA\\\\BB\\\\CC.txt）：")
#         status = int(input("采用庞双子（2021）分析框架请输入“1”，采用朱晓楠（2015）分析框架请输入“2”，结束使用请输入“0”："))
#         pos_path = pos_output(txt_folder_path)
#         if status == 1:
#             MDC_result_output(pos_path,'pang')
#         elif status == 2:
#             MDC_result_output(pos_path,'zhu')
#         else:
#             break
#         # try:
#         #     csv_to_txt(csv_path)
#         # except Exception as e:
#         #     if any(s in e for s in ["filename","'utf-8' codec can't decode"]):
#         #         print('本工具目前仅支持UTF-8编码的csv文件转换，请修改您的csv文件编码后重试！')
#         #     elif 'Invalid argument' in e:
#         #         print('输入的绝对路径不合规范，请确认绝对路径无误后重试！')
#         #     elif 'The system cannot find the path specified' in e:
#         #         print('csv的filename列中存在不合规范的路径，请确认此列内容无误后重试！')
#         #     else:
#         #         print("except",e)
#     # elif command == 3:
#     #     source_folder_path = input("请输入待拆分的文件夹的绝对路径（示例格式为：C:\\\\Users\\\\22961\\\\Documents\\\\讯飞\\\\0718\\\\文件夹名）：")
#     #     n = int(input("您预期将此文件夹拆分为多少个子文件夹？请注意输入数字必须为整数："))
#     #     try:
#     #         file_move(source_folder_path,n)
#     #     except Exception as e:
#     #         print(e)
#     elif command == 0:
#         break
#     else:
#         print("指令不合规范，请重新输入指令！")
#     turn_on_or_off = input("继续使用本工具请输入“0”以外任意字符，结束使用请输入“0”：")
# print("程序结束！")