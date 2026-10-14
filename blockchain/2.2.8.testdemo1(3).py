from blockchain import Blockchain
import json
import time
from hashlib import sha256
from ellipticcurve.ecdsa import Ecdsa
from ellipticcurve.privateKey import PrivateKey

def create_accounts():
    accounts=[]
    for i in range(10):
        seed=str(time.time())
        time.sleep(0.001)
        account=sha256(seed.encode()).hexdigest()
        accounts.append(account)
    return accounts

privateKey = PrivateKey()
publicKey = privateKey.publicKey()


accounts=create_accounts()

blockchain=Blockchain()

blockchain.create_genesis_block()

sender=accounts[0]
receiver=accounts[1]
data="My first transaction testing data."

ret=blockchain.send_transaction(sender,receiver,data)

sender=accounts[2]
receiver=accounts[5]
data="My 2nd transaction testing data."

ret=blockchain.send_transaction(sender,receiver,data)

blockchain.mine()

sender=accounts[3]
receiver=accounts[7]
data="My 3rd transaction testing data."

ret=blockchain.send_transaction(sender,receiver,data)
blockchain.mine()

'''
sender=accounts[3]
receiver=accounts[7]
data="My 4th transaction testing data."

signature=blockchain._send_transaction(sender,receiver,data,privateKey)
blockchain._mine(publicKey,signature,data)
'''


print("Height of Blockchain:",blockchain.get_height())
print("Detailed Information of The 2nd Block:\n",blockchain.get_block(2))
print("The Last Block:\n",blockchain.last_block.__dict__)
chain=blockchain.get_chain()
print("Blockchain Information:\n",chain)

