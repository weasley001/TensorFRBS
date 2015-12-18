import argparse, os, sys,time,itertools
from sklearn import cluster, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#this version accept different number of rules for each class
class TensorFuzzyRuleBase:

    def __init__(self,numofclass,numofrulesperclass):
        self.nc = numofclass
        self.nrpc = numofrulesperclass
        self.nr = np.sum(self.nrpc)
        self.cnrpc = np.cumsum(np.concatenate(([0],self.nrpc),axis = 0))
        
        
    
    def fit(self, data, target):
        self.dim = data.shape[-1]
        self.X = (map(lambda i: data[target == i],xrange(self.nc)))
        #[[0,1,2],[3,4],[5,6,7,8]]
        self.rule_target = np.asarray( map(lambda x,y: np.asarray(range(x,y)) , self.cnrpc[0:-1],self.cnrpc[1:]))
        #{0:(0,0),1:(0,1)}
        self.rule_mapping ={c:(d,e) for c,(d,e) in zip(xrange(self.nr), self.myrange(self.nrpc))}
        #print(self.rule_mapping) 
        self.s_min = 0.0001
        Y = (map(self.getcluster, xrange(self.nc)))
        print("Clustering")
        self.segment_ids = [_k for _i,_j in enumerate(self.nrpc) for _k in itertools.repeat(_i,_j)]
        #print(self.segment_ids)
        #[0,0,0,1,1,1,2,2,2,]
        ##[(i,j) for i,k in enumerate(x) for j in xrange(k)] = my range2
        ##[k for i,j in enumerate(x) for k in intertools.repeate(i,j)]
        M = np.asarray(map(lambda (i,j):self.X[i][Y[i][:,j]>=0.5].mean(axis=0),TensorFuzzyRuleBase.myrange(self.nrpc)))
        S = self.s_min + np.asarray(map(lambda (i,j):self.X[i][Y[i][:,j]>=0.5].std(axis=0),TensorFuzzyRuleBase.myrange(self.nrpc)))
        self.m = M
        #self.m += 0.0*np.random.rand(self.nr,self.dim)
        self.s = S
        self.s = np.ones((self.nr,self.dim))+0.0 
        self.s2 = np.square(self.s)
        return 0

    def droprule(self, drop_index):
        k_class,l_rule = drop_index
        if (self.nrpc[k_class] < 2):
            print(("One rule in %d class, No dropout")%(k_class))
            return False
        elif (self.nrpc[k_class] < l_rule):
            print(("No rule %d in %d class, No dropout") % (l_rule,k_class))
            return False
        else:
            self.m = np.delete(self.m,self.rule_target[k_class][l_rule], 0)
            self.s = np.delete(self.s,self.rule_target[k_class][l_rule], 0)
            self.s2 = np.delete(self.s2,self.rule_target[k_class][l_rule], 0)
            
            self.nrpc[k_class] -= 1
            self.nr = np.sum(self.nrpc)
            self.cnrpc = np.cumsum(np.concatenate(([0],self.nrpc),axis = 0))
            self.rule_target = np.asarray( map(lambda x,y: np.asarray(range(x,y)) , self.cnrpc[0:-1],self.cnrpc[1:]))
            self.rule_mapping ={c:(d,e) for c,(d,e) in zip(xrange(self.nr), self.myrange(self.nrpc))}
            #print(self.rule_target)
            return True
    
    def droprules(self, drop_list):
        #drop_list = [7,3,5,9,1]
        rv = drop_list[:]
        rv.reverse()
        while rv:
            i = rv.pop()
            if self.droprule(self.rule_mapping[i]):
                rv = [x if x < i else x-1 for x in rv ]
                


    def getcluster(self,k):

        if len(self.X[k]) < self.nrpc[k] :

            k_means = cluster.KMeans(n_clusters = len(self.X[k]))
            k_means.fit(self.X[k])
            label = np.zeros((len(self.X[k]), self.nrpc[k]))
            map(lambda i,j: label.itemset((i,j),1),xrange(len(self.X[k])),k_means.labels_)
            label[:,range(len(self.X[k]),self.nrpc[k])] = 1
            print("Warring:Rule number is larger than instance number in Class %d" %(k))
        else:
            k_means = cluster.KMeans(n_clusters = self.nrpc[k],max_iter = 10000,init= 'random',n_init = 30)
            k_means.fit(self.X[k])
            label = np.zeros((len(self.X[k]), self.nrpc[k]))
            map(lambda i,j: label.itemset((i,j),1),xrange(len(self.X[k])),k_means.labels_)
        return label
    
    def gaussianMF(self,datapoint):
        
        mfv = np.subtract(datapoint,self.m)
        np.square(mfv,mfv)
        np.negative(mfv,mfv)
        np.divide(mfv,self.s2,mfv)
        np.exp(mfv,mfv)
        
        #mfv = np.exp(np.divide(np.negative(np.square(np.subtract(datapoint,self.m))),np.square(self.s)))
        #inline version maybe slower
        
        return mfv               
    
    def getFS(self,datapoint):
        fss = np.min(self.gaussianMF(datapoint), axis=1)
        fss = np.asarray(map(lambda x:(fss[self.rule_target[x]]).max(),xrange(self.nc)))
        return fss

    def getsoftFS(self,datapoint):
        fss = np.prod(self.gaussianMF(datapoint), axis=1)
        fss = np.asarray(map(lambda x:self.softMINMAX(fss[self.rule_target[x]],50),xrange(self.nc)))
        return fss
    
    def getEffectedRule(self,datapoints,ans):
        fss = np.zeros(self.nr)
        err = np.zeros(self.nr)
        
        for i,datapoint in enumerate(datapoints):
            maxr = np.argmax(np.prod(self.gaussianMF(datapoint), axis=1))
            fss[maxr] += 1
            if self.rule_mapping[maxr][0] == ans[i]:
                err[maxr] += 1
     
        return fss,err
    
    def shink(self, datapoints, data_target):
        hits, corr = self.getEffectedRule(datapoints,data_target)
        err = hits - corr
        class_size = map(lambda i: len(data_target[data_target == i]),xrange(self.nc))
        
        del_list = set()
        for i in xrange(self.nr):
            k_class, l_rule = self.rule_mapping[i]
            if (err[i] > corr[i]):
                del_list.add(i)
            elif hits[i] <= (class_size[k_class] * 0.5 / self.nrpc[k_class]):
                del_list.add(i)    
        del_list = list(del_list)
        print(map(lambda x:self.rule_mapping[x],del_list))
        self.droprules(del_list)    
        
    def predict(self,datapoint):
        return np.argmax(self.getFS(datapoint),axis = 0)
    
    def tune(self,datapoints,data_target):
        learning_rate = 0.01
        training_epochs = 200
        display_step = 50
        X_train = datapoints
        Y_train = np.zeros((len(X_train),self.nc))
        for _i,_x in enumerate(data_target):
            Y_train[_i,_x] = 1.0

        _m = tf.Variable(np.float32(self.m))
        _s = tf.Variable(np.float32(self.s))
        x = tf.placeholder("float32", shape=[self.dim])
        y = tf.placeholder("float32", shape=[self.nc])
        gmf = tf.exp(-1*(tf.square(tf.div(tf.sub(x,_m),_s))))
        fire_strength = tf.reduce_min(gmf,1)
        activation = tf.segment_max(fire_strength,self.segment_ids)
        loss = tf.reduce_sum(tf.square(tf.sub(y,activation)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list =[_m,_s])
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            total_error = 0.
            
            for epoch in xrange(training_epochs):
                total_error = 0
                start_time = time.clock()
                for i in range(len(X_train)):
                    batch_xs, batch_ys = X_train[i],Y_train[i]
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                    total_error += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
                    #if i % display_step == 1:
                        #print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(total_error)
                    #print (m.eval(sess))
                    #print (s.eval(sess))
                end_time = time.clock() 
                if epoch % display_step == 0:   
                    print "time this epoch=", (end_time-start_time)   
                    print "Epoch:", '%04d' % (epoch +1), "cost=", "{:.9f}".format((total_error/len(X_train)))
            self.m = sess.run(_m)
            self.s = sess.run(_s)
            self.s2 = np.square(self.s)
        


    def printRules(self,rs_value,hitrules,fout):
        with open(fout,'w') as fo:
            rule_class = self.myrange(self.nrpc)

            for i,j in enumerate(zip(rs_value,rule_class)):
                #fo.write("%d\t"%(j[1][1]))
                for m,n in zip(self.m[i],self.s[i]):
                    fo.write("(%.3f,%.3f)\t"%(m,n))
                fo.write("%d\t%d\t%d\t%d\n"%(j[0],hitrules[i],j[1][0],j[1][1]))  

    @staticmethod
    def myrange(i):
        for _i,_j in enumerate(i):
            for _k in xrange(_j):
                yield (_i,_k)            
    @staticmethod
    def softMINMAX(alpha,q):
        denominator = np.exp(alpha*q)
        numerator = alpha*denominator
        return numerator.sum()/denominator.sum()
         
   
def err_handler(type, flag):
    print "Floating point error (%s), with flag %s" % (type, flag)
    #traceback.print_stack()

def plotfig(fout):
    #x = np.linspace(scaler.mean_.min()-2*scaler.std_.max(),scaler.mean_.max()+2*scaler.std_.max(),1024)
    x = np.linspace(-4,4,1024)
    xs = np.linspace(scaler.mean_.min()-2*scaler.scale_.max(),scaler.mean_.max()+2*scaler.scale_.max(),128)
    gs = plt.GridSpec(np.asarray(R1.nrpc).max()+1, R1.nc)
    plt.subplot(gs[0,:])

    for k in xrange(R1.dim):
        density = gaussian_kde(raw_data[:,k])
        density.covariance_factor = lambda : .20
        density._compute_covariance()
        plt.plot(xs,density(xs),label=''+df_feature_name[k])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=R1.dim, mode="expand", borderaxespad=0.)
    #R1.cnrpc.tolist()
    for c,(d,e) in zip(xrange(R1.nr), R1.myrange(R1.nrpc)):
        #print (c,(d,e))
        plt.subplot(gs[e+1,d])
        for i,j,k in zip(R1.m[c],R1.s[c],xrange(R1.dim)):
        #for i,j,k in zip(R1.m[c]+scaler.mean_,R1.s[c]*scaler.std_,xrange(R1.dim)):
            plt.plot(x,np.exp(-(x-i)**2/(j**2)),label=''+df_feature_name[k])
        #plt.legend() 
    plt.savefig('fig'+str(fout))
    plt.close()  
    #plt.show()
    return plt

def plotfig2(fout):
    #x = np.linspace(scaler.mean_.min()-2*scaler.std_.max(),scaler.mean_.max()+2*scaler.std_.max(),1024)
    x = np.linspace(-4,4,1024)
    #xs = np.linspace(scaler.mean_.min()-2*scaler.scale_.max(),scaler.mean_.max()+2*scaler.scale_.max(),128)
    xs = np.linspace(scaler.mean_.min()-2*scaler.std_.max(),scaler.mean_.max()+2*scaler.std_.max(),128)
    gs = plt.GridSpec(R1.dim+1, 1)
    plt.subplot(gs[0,:])

    for k in xrange(R1.dim):
        density = gaussian_kde(raw_data[:,k])
        density.covariance_factor = lambda : .20
        density._compute_covariance()
        plt.plot(xs,density(xs),label=''+df_feature_name[k])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=R1.dim, mode="expand", borderaxespad=0.)
    #R1.cnrpc.tolist()

    for c in xrange(1,R1.dim+1):
        plt.subplot(gs[c,:])
        for i,j,k in zip(R1.m[:,c-1],R1.s[:,c-1],xrange(1,R1.nr+1)):
            plt.plot(x,np.exp(-(x-i)**2/(j**2)),label='F'+str(c)+' Set'+str(k))
        plt.legend()
       
        
    plt.savefig('fig'+str(fout))
    plt.close()  
    #plt.show()
    return plt


if __name__ == '__main__':
    ################################################################################
    #setting argument
    saved_handler = np.seterrcall (err_handler)
    old_settings = np.seterr(all='call',under="ignore")
    #old_settings = np.seterr(all='warn', over='raise')
    parser = argparse.ArgumentParser(description='simple FRBS demo program for classification problem') 
    parser.add_argument('-f',required = True)
    parser.add_argument('-k',type=int,default = 3)
    args = parser.parse_args()
    infile_path = args.f
    rulesperclass = args.k
    ################################################################################
    #read input training file and show some information
    print("input file: %s"% (os.path.basename(infile_path)))
    
    df = pd.read_csv(infile_path, sep=',')
    df = df.dropna(axis = 1,how='all')
    target_column = ''
    if df.filter(axis=1, regex=('CLASS')).shape[1] == 1:
        target_column = df.filter(axis=1, regex=('CLASS')).columns[0]
    else:
        target_column = df.columns[-1]
    #print(target_column)    
    
    df_feature_name = df.drop(target_column,axis=1).columns.values
    raw_data = df.drop(target_column, axis=1).values
    data_target_names = df[target_column].unique()
    ################################################################################
    #maping category class label to numerical
    target_mapping = dict()
    for i,j in enumerate(data_target_names):
        target_mapping[j] = i
    data_target = np.asarray(map(lambda x: target_mapping.get(x) if target_mapping.has_key(x) else -1,df[target_column]))   
    print(target_mapping)
    classsize = map(lambda i: len(data_target[data_target == i]),xrange(len(data_target_names)))
    print(classsize)
    #rulesperclass = [int(i/8.0) if i > args.k*8.0 else args.k for i in classsize]
    #rulesperclass = [1,2,2]
    rulesperclass = [args.k] * len(data_target_names)
    

    print("rules per class: ", rulesperclass)
    print("number of class: %d"% (len(data_target_names)))
    print("number of fuzzy rules: %d"% (np.asarray(rulesperclass).sum()))
    print("number of instance: %d"% (len(df.index)))
    print("-"*80)
    ################################################################################
    #Scaling inupt data set
    scaler = preprocessing.StandardScaler().fit(raw_data)
    np.set_printoptions(precision=2,suppress=False,formatter={'float': '{: 0.3f}'.format})
    #print(scaler.mean_)
    #print(scaler.std_)

    sys.stdout.flush()
    data_scaled = scaler.transform(raw_data)
    print(len(data_scaled))
    print(data_target)
    

    
    #np.savetxt("scaled.csv", data_scaled , delimiter=",")
    ################################################################################
    #build fuzzy system and predict
    #X_train, X_test, Y_train, Y_test = train_test_split(data_scaled, data_target, test_size=0.10, random_state=2501)
    k_fold = StratifiedKFold(data_target, n_folds=3,shuffle=True)
    my_ans = list()
    k_ans = list()
    for train_index, test_index in k_fold:
        X_train = data_scaled[train_index]
        Y_train = data_target[train_index]
        X_test = data_scaled[test_index]
        Y_test = data_target[test_index]
        R1 = TensorFuzzyRuleBase(len(data_target_names),rulesperclass)
        R1.fit(X_train,Y_train)

         
        cm = confusion_matrix(Y_train, map(R1.predict,X_train))
        print("Training Accuracy:%2.4f" % (np.trace(cm)/(cm.sum()+0.0)))
        #print(R1.s2)
        #optimization done by tensorflow
        R1.tune(X_train,Y_train)
        #print(R1.s2)
         
        cm = confusion_matrix(Y_train, map(R1.predict,X_train))
        print("Tuned Training Accuracy:%2.4f" % (np.trace(cm)/(cm.sum()+0.0)))

        my_ans.extend(map(R1.predict,X_test))
        k_ans.extend(Y_test)

    cm = confusion_matrix(k_ans, my_ans)
    print("Confusion Matrix")
    print(cm)
    print("-"*80)
    print("Accuracy:%2.4f" % (np.trace(cm)/(cm.sum()+0.0)))
       
    sys.stdout.flush()
       
    
   
