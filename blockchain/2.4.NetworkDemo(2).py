import random
import json
from blockchain import Blockchain
from walletdemo import Wallet

blockchain=Blockchain()

class Network(object):

    def __init__(self):        
        self.peers = set()
        self.off_peers = []  

    def create_genesis_block(self):        
        genesis_block = Block(0, [], 0, "0")
        genesis_block.hash = genesis_block._calculate_hash()
        self.chain.append(genesis_block)
        return genesis_block 
    
    def add_peer(self, host,port):
        self.host=host
        self.port=port
        peer=str(host)+":"+str(port)               
        self.peers.add(peer)        
        return str(self.peers)

    def login(self, peer):
        if peer in self.peers:
            print(str(peer.peerName)+" was already logged in!")            
        else:
            print("Peer " + str(peer.peerName) + " connected to the P2P network.")        
            self.peers.add(peer)
            peer.online=True        
            return str(self.peers)
    
    def logout(self,peer):
        if len(self.peers)<1:
            print("No Peers on the network.")
            return        
        if not peer in self.peers:
            print("Peer " + str(peer.peerName) + " was NOT logged in yet!")
        else:
            self.peers.remove(peer)
            self.off_peers.append(peer)
            peer.online=False
            print("Peer "+  str(peer.peerName)  + " logged out.")
        return self.off_peers        
    
    def peer_online(self):
        peer_online=[]
        for peer in network.peers:
            peer_online.append(peer.peerName)
        print("在线节点：",peer_online)
        return peer_online
    
    def create_peer(self,peer):
        peer.pid = self.nop        
        peer.wallet.generate_keys()
        self.peers.append(peer)
    
    def generate_random_peer(self,n):
        host='127.0.0.1'
        for i in range(5):
            port=random.randint(5001,9000)        
            self.add_peer(host,port)
        return self.peers    
    

class Peer:
    def __init__(self,name=None):  #节点初始化函数
        self.online = False  
        self.peerName=name
        self.chain=[]
        self._is_wallet_generated = False
        self.generate_wallet()
    
    def generate_wallet(self): #用于节点生成钱包对象
        if not self._is_wallet_generated:
            self.wallet = Wallet()
            self._is_wallet_generated = True

    '''
    节点区块链同步函数。
    用于节点与网络之间的区块链进行数据同步。
    '''
    def syn_blockchain(self,blockchain): 
        if len(self.chain)<blockchain.get_height():
            chaindata=blockchain.get_chain()            
            for block in chaindata:                
                self.chain.append(block)
        return self.chain         
    
    def print_blockchain(self): #输出节点的区块链副本数据
        print("本节点区块链包含区块个数: %d" % len(self.chain))
        for block in self.chain:            
            print("区块索引：%s" % block['index'])
            print("上个区块哈希：%s" % block['previous_hash'])
            print("区块交易信息：%s" % block['transactions'])
            print("本区块哈希：%s" % block['hash'])
            print("===========================================================================\n")  

    '''
    节点版本的交易函数。
    用于节点之间发送交易数据。交易需要后续挖矿来确认从而产生新的区块。
    '''
    def send_transaction(self,sender,receiver,tx_data): 
        if sender.online is False: #若节点不在线则不可发送交易
            print("Peer NOT Logged in yet! Transaction Abort!")
        else:            
            sender=sender.wallet.address.decode() #发送节点的钱包地址作为交易地址
            receiver=receiver.wallet.address.decode() #接收节点的钱包地址作为交易地址
            blockchain.send_transaction(sender,receiver,tx_data)
            print("Transaction was sent successfully by Peer!")




if __name__ == "__main__":
    network=Network() #网络对象初始化
        
    blockchain.create_genesis_block() #生成创世区块
    
    polaris=Peer("Polaris") #注册新节点
    polaris_wallet=polaris.wallet  #新节点钱包对象
    prcaurora=Peer("Prcaurora")
    prcaurora_wallet=prcaurora.wallet     
    
    #通过网络版本的交易函数发送第1个交易并挖矿确认
    sender=polaris_wallet.address.decode()  
    receiver=prcaurora_wallet.address.decode()
    data="My 1st transaction testing data."
    ret=blockchain.send_transaction(sender,receiver,data)
    blockchain.mine() #挖矿确认交易并产生一个新区块
    
    #通过网络版本的交易函数发送第2个交易并挖矿确认
    receiver=polaris_wallet.address.decode()  
    sender=prcaurora_wallet.address.decode()
    data="My 2nd transaction testing data."
    ret=blockchain.send_transaction(sender,receiver,data)
    blockchain.mine()
    

    network.login(polaris)     #节点polaris登录
    network.login(prcaurora)   #节点prcaurora登录
    network.peer_online()      #查询在线节点
    network.logout(prcaurora)  #节点prcaurora注销 
    network.peer_online()      #查询在线节点
    
    #节点prcaurora通过节点版本的交易函数发送1笔交易，应该会失败，应为该节点已经注销，处于不在线状态。
    prcaurora.send_transaction(prcaurora,polaris,"Testing message") 
    #节点polaris通过节点版本的交易函数发送1笔交易    
    polaris.send_transaction(polaris,prcaurora,"Testing message")     
    blockchain.mine()  #确认交易并产生新区块

    
    res=polaris.syn_blockchain(blockchain) #节点区块链同步
    print("已同步节点polaris区块链信息：")  
    polaris.print_blockchain()             #输出该节点的区块链信息
    print("JSON格式区块链信息：\n")
    print(json.dumps(res,indent=3))        #输出该节点JSON格式的区块链信息
    print("未同步节点prcaurora区块链信息：")
    prcaurora.print_blockchain()           #prcaurora节点未进行区块链数据同步，因此将输出新节点为0
    
