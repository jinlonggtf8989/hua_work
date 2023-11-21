#散点图
# import matplotlib.pyplot as plt
# from pylab import *
# mpl.rcParams['font.sans-serif']=['SimHei']
# mpl.rcParams['axes.unicode_minus']=False
# x_values=list(range(1,1001))
# y_values=[x**2 for x in x_values]
# plt.scatter(x_values,y_values,c=(0,0,0.8),edgecolors='red',s=40)
# plt.title('销售表',fontsize=24)
# plt.xlabel('节点',fontsize=14)
# plt.ylabel('销售数据',fontsize=14)
# plt.tick_params(axis='both',which='major',labelsize=14)
# plt.axis([0,110,0,1100])
# plt.show()

# x=[1,2]
# y=[2,4]
# plt.scatter(x,y)
# plt.axis([0,10,0,10])
# plt.show()

# x_values=range(1,6)
# y_values=[x*x for x in x_values]
# plt.scatter(x_values,y_values,s=50)
# plt.title('scatter of beautiful',fontsize=24)
# plt.xlabel('value',fontsize=14)
# plt.ylabel('square of value',fontsize=14)
# plt.tick_params(axis='both',which='major',labelsize=14)
# plt.show()

# import numpy as np
# N=10
# x=np.random.rand(N)
# y=np.random.rand(N)
# s=(30*np.random.rand(N))**2
# c=np.random.rand(N)
# plt.title('compond picture')
# plt.xlabel('value',fontsize=14)
# plt.ylabel('y_value',fontsize=14)
# plt.axis([0,100,0,100])
# plt.scatter(x,y,s=s,c=c,alpha=0.5,marker='^',label='random picture')
# plt.legend(loc='best')
# plt.show()
#折线图

# import matplotlib.pyplot as plt
# x_values=range(1,1001)
# y_values=[x*x for x in x_values]
# plt.scatter(x_values,y_values,c=y_values,cma
# p=plt.cm.Blues,edgecolors='none',s=10)
# plt.title('square of Numbers',fontsize=24)
# plt.xlabel('value',fontsize=14)
# plt.ylabel('Square of value',fontsize=14)
# plt.tick_params(axis='both',which='major',labelsize=14)
# plt.axis([0,1100,0,1100000])
# plt.show() plt.plot(x_values,y_values,linewidth=5)

# 绘制多幅子图

# import matplotlib.pyplot as plt
# import numpy as np

# pylab库简便了程序
# from pylab import *

# X=np.linspace(-np.pi,np.pi,256,endpoint=True)
# C,S=np.cos(X),np.sin(X)
# plot(X,C)
# plot(X,S)
# show()

# 子图
# fig=plt.figure()
# p1=fig.add_subplot(211)
# x=list(range(1,8))
# y=np.random.rand(7)
# p1.plot(x,y)
# p2=fig.add_subplot(212)
# a=[1,2]
# b=[2,4]
# p2.scatter(a,b)
# plt.show() 

# 对一些默认设置进行改变
# from pylab import *
# figure(figsize=(8,6),dpi=80)
# subplot(1,1,1)
# X=np.linspace(-np.pi,np.pi,256,endpoint=True)
# C,S=np.cos(X),np.sin(X)
# plot(X,C,color='blue',linewidth=1.0,linestyle='-')
# plot(X,S,color='green',linewidth=1.0,linestyle='-')
# xlim(-4.0,4.0)
# ylim(-1.0,1.0)
# xticks(np.linspace(-4,4,9,endpoint=True))
# yticks(np.linspace(-1,1,5,endpoint=True))
# text(1,1,'sin(x)=x')
# savefig('exercise_2.png',dpi=72)
# show()

#plot可绘制多个不同类型图
# import numpy as np
# import matplotlib.pyplot as plt

# t=np.arange(0.,5.,0.2)
# plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
# plt.show()


