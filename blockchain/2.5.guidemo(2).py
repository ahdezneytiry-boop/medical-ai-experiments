import json
import random
from logger import logger
from blockchain import Blockchain
from walletdemo import Wallet
from NetworkDemo import Network,Peer

import tkinter
from tkinter import *
from tkinter import ttk  #导入ttk模块，因为Combobox下拉菜单控件在ttk中
root = Tk()
root.title("区块链模拟系统 V1.0 @ J.T.Lu")


peers_registered=set()
peers_online=set()
peers_offline=[]

network=Network()
wallet=Wallet()
blockchain=Blockchain()
blockchain.create_genesis_block()

def MenuCallback1():    
    logger.info("系统初始化,生成随机节点...")
    blockchain_info_Text.insert(INSERT,"系统初始化,生成5个随机节点...")
    blockchain_info_Text.insert(INSERT,"\n")     
    for _ in range(5):
        node=str(random.randint(1,254))+"."+str(random.randint(1,254))+"."+str(random.randint(1,254))+"."+str(random.randint(1,254))
        logger.info(node)
        peers_online.add(node)
        peers_registered.add(node) 
        blockchain_info_Text.insert(INSERT,node)
        blockchain_info_Text.insert(INSERT,"\n")    


    #刷新在线节点列表值
    online_peers_set=[i for i in peers_online]    
    online_combobox['value'] = online_peers_set
    online_combobox.current(0)
    
    offline_peers_set=[i for i in peers_offline]
    offline_combobox['value'] = offline_peers_set 
    register_peer_text.delete(0,'end')   

    #交易地址控件赋值
    tx_address=[str(Peer(w).peerName)+ ":"+str(Peer(w).wallet.address.decode()) for w in peers_registered]
    sender_peer['value']=tx_address    
    receiver_peer['value']=tx_address  
    sender_peer.current(0)
    receiver_peer.current(0)  
    

def MenuCallback2():
    logger.info("退出系统...") 
    exit(0)

def MenuCallback3():
    print("日志显示")


def peer_register(): #节点注册
    peerName=register_peer_text.get()
    peers_online.add(peerName)
    peers_registered.add(peerName)
    info1="Peer "+ peerName +" was Registered!"
    logger.info(info1)    
    info2="Current online peers:"+str(peers_online)
    logger.info(info2)
    blockchain_info_Text.insert(INSERT,info1)
    blockchain_info_Text.insert(INSERT,"\n") 
    blockchain_info_Text.insert(INSERT,info2)
    blockchain_info_Text.insert(INSERT,"\n") 
    
    #刷新在线节点列表值
    online_peers_set=[i for i in peers_online]    
    online_combobox['value'] = online_peers_set
    online_combobox.current(0)
    offline_peers_set=[i for i in peers_offline]
    offline_combobox['value'] = offline_peers_set 
    offline_combobox.current(0)
    register_peer_text.delete(0,'end')   

    #交易地址控件赋值
    tx_address=[str(Peer(w).peerName)+ ":"+str(Peer(w).wallet.address.decode()) for w in peers_registered]
    sender_peer['value']=tx_address    
    receiver_peer['value']=tx_address
    

def peer_login(): #节点登录
    peername=var_offline_peer.get()
    info="Peer " + peername + " Connected to the network."
    print(info)
    blockchain_info_Text.insert(INSERT,info)
    blockchain_info_Text.insert(INSERT,"\n")  

    peers_online.add(peername)
    peers_offline.remove(peername)

    online_peers_set=[i for i in peers_online]    
    online_combobox['value'] = online_peers_set
    online_combobox.current(0)
    offline_peers_set=[i for i in peers_offline]
    offline_combobox['value'] = offline_peers_set
    offline_combobox.set("")
    offline_combobox.current(0)

def peer_logout(): #节点注销
    peername=var_online_peer.get()
    info="Peer " + peername + " logged off."
    print(info)
    blockchain_info_Text.insert(INSERT,info)
    blockchain_info_Text.insert(INSERT,"\n")  
    peers_offline.append(peername)
    peers_online.remove(peername)

    online_peers_set=[i for i in peers_online]    
    online_combobox['value'] = online_peers_set
    online_combobox.set("") 
    online_combobox.current(0)
    offline_peers_set=[i for i in peers_offline]
    offline_combobox['value'] = offline_peers_set  
    offline_combobox.current(0)  

def peer_send_tx():
    sender_name,sender_address=str(sender_peer.get()).split(":")    
    receiver_name,receiver_address=str(receiver_peer.get()).split(":")
    data=tx_data.get()  
    blockchain.send_transaction(sender_address,receiver_address,data) 
    
    info="Transaction was sent. From:"+sender_name+" To:"+receiver_name
    blockchain_info_Text.insert(INSERT,info)
    blockchain_info_Text.insert(INSERT,"\n") 

def print_unconfirmed_tx():
    blockchain_info_Text.insert(INSERT,"未确认交易数据:")
    blockchain_info_Text.insert(INSERT,"\n")
    for i in blockchain.unconfirmed_transactions: 
        blockchain_info_Text.insert(INSERT,i)
        blockchain_info_Text.insert(INSERT,"\n") 

def peer_tx_mine():
    num_tx_to_mine=len(blockchain.unconfirmed_transactions)
    ret=blockchain.mine()
    if ret:
        logger.info("%s transactions were mined!" %num_tx_to_mine)
        info=str(num_tx_to_mine) +" transactions were mined!"
        info1="New Block added into the blockchain."
        blockchain_info_Text.insert(INSERT,info)
        blockchain_info_Text.insert(INSERT,"\n") 
        blockchain_info_Text.insert(INSERT,info1)
        blockchain_info_Text.insert(INSERT,"\n") 

def show_block_information():    
    index=block_index_data.get()
    if index=="last":
        ret=blockchain.last_block
        logger.info(ret.__dict__)
    else:
        index=int(index)
        if index>=blockchain.get_height():            
            blockchain_info_Text.insert(INSERT,"Index exceed the height of the blockchain!")
            blockchain_info_Text.insert(INSERT,"\n") 
            logger.info("Index exceed the height of the blockchain!")
            return
        else:            
            ret=blockchain.get_block(index)
            blockchain_info_Text.insert(INSERT,ret)
            blockchain_info_Text.insert(INSERT,"\n")
            logger.info(ret)    
    return ret

def get_blockchain_height():
    info="当前区块链高度:%s"%blockchain.get_height()
    logger.info(info)
    blockchain_info_Text.insert(INSERT,info)
    blockchain_info_Text.insert(INSERT,"\n") 
    return blockchain.get_height()

def get_blockchain():
    peerName=online_combobox.get()
    peer=Peer(peerName)    
    peer.syn_blockchain(blockchain)    
    logger.info(peer.print_blockchain())
    
def show(event):
    online_peers_set=[i for i in peers_online]    
    online_combobox['value'] = online_peers_set
    offline_peers_set=[i for i in peers_offline]    
    offline_combobox['value'] = offline_peers_set
    

menubar = Menu(root)  #创建一个顶级菜单
filemenu = Menu(menubar,tearoff = False) #创建一个子菜单filemenu，长在顶级菜单menubar下
menubar.add_cascade(label = "系统",menu = filemenu) #为子菜单filemenu取个名字,长在顶级菜单menubar下
filemenu.add_command(label = "节点初始化",command = MenuCallback1) #为子菜单filemenu添加选项，取名"另存为"
#filemenu.add_command(label = "日志文件",command = MenuCallback3)
filemenu.add_command(label = "退出系统",command = MenuCallback2)  #为子菜单filemenu添加选项，取名"关闭"

root.config(menu = menubar)


var_online_peer=tkinter.StringVar()
var_offline_peer=tkinter.StringVar()

function_op_grp=LabelFrame(root,text="功能区",relief=SUNKEN,borderwidth=2)
function_op_grp.grid(row=0,column=1)

peer_management_set = LabelFrame(function_op_grp,text = "网络节点管理",padx=5,pady=5,relief=RAISED)
peer_management_set.grid(row = 0,column = 1,sticky=W)


online_label = Label(peer_management_set,text = "在线节点",foreground="black")
online_label.grid(row = 1,column = 1,sticky = W)
online_combobox = ttk.Combobox(peer_management_set,textvariable=var_online_peer,width = 11)
online_combobox.bind('<<ComboboxSelected>>', show)

online_peers_set=[i for i in peers_online]
online_combobox['value'] = online_peers_set
online_combobox.grid(row = 1,column = 2)
logout_button=Button(peer_management_set,text="节点注销",command=peer_logout)
logout_button.grid(row = 1, column = 3)

offline_label = Label(peer_management_set,text = "离线节点")
offline_label.grid(row = 2,column = 1)
offline_combobox = ttk.Combobox(peer_management_set,textvariable=var_offline_peer,width = 11)

offline_peers_set=[i for i in peers_offline]
offline_combobox['value'] = offline_peers_set
offline_combobox.grid(row = 2,column = 2)
login_button=Button(peer_management_set,text="节点登录",command=peer_login)
login_button.grid(row = 2, column = 3)

#inputBox_val=tkinter.StringVar()
register_peer_text = Entry(peer_management_set,width = 14)
register_peer_text.grid(row = 3,column = 2,sticky=W)
register_button = Button(peer_management_set,text = "节点注册",command=peer_register)
register_button.grid(row = 3,column = 3,sticky=E)

#交易相关GUI
transaction_frame=LabelFrame(function_op_grp,text="交易管理",padx=5,pady=5,relief=RAISED)
transaction_frame.grid(row = 1,column = 1,sticky=W)

Label(transaction_frame,text = "发送地址").grid(row=1,column=1,sticky=W)
sender_peer = ttk.Combobox(transaction_frame,width =19)
sender_peer.grid(row = 1,column = 2)
Label(transaction_frame,text = "接收地址").grid(row=2,column=1,sticky=W)
receiver_peer= ttk.Combobox(transaction_frame,width = 19)
receiver_peer.grid(row = 2,column = 2)
Label(transaction_frame,text = "交易数据").grid(row=3,column=1,sticky=W)
tx_data= Entry(transaction_frame,width=21)
tx_data.grid(row = 3,column = 2)
peer_send_tx = Button(transaction_frame,text = "发送交易",command=peer_send_tx)
peer_send_tx.grid(row = 4,column = 2,sticky=E)
Button(transaction_frame,text = "数据上链",command=peer_tx_mine).grid(row =5,column = 1,sticky=W)
Button(transaction_frame,text = "未确认交易池",command=print_unconfirmed_tx).grid(row =5,column = 2,sticky=E)

#区块链相关操作
blockchain_op = LabelFrame(function_op_grp,text = "区块与区块链",padx=5,pady=5,relief=RAISED)
blockchain_op.grid(row=2,column=1,sticky=W)
Label(blockchain_op,text = "区块索引",foreground="black").grid(row=0,column=1,sticky=W)
block_index_data=Entry(blockchain_op,width=10)
block_index_data.grid(row=0,column=2,sticky=W) 
Button(blockchain_op,text = "区块信息",command=show_block_information).grid(row =0,column = 3,sticky=E)
Button(blockchain_op,text = "区块链高度",command=get_blockchain_height).grid(row =1,column = 1,sticky=W)
Button(blockchain_op,text = "区块链信息",command=get_blockchain).grid(row =1,column = 3,sticky=W)


#信息显示区域
display_info_grp = LabelFrame(root,width = 800,text="综合信息显示区") 
display_info_grp.grid(row = 0,column = 2,rowspan=3)
blockchain_info_Text=Text(display_info_grp,width = 60,height = 28,pady = 5)
blockchain_info_Text.grid(row = 0,column = 1)

mainloop()
