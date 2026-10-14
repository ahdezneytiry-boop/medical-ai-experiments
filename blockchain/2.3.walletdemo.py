#导入椭圆曲线算法
from ecdsa import SigningKey, SECP256k1, VerifyingKey, BadSignatureError
import binascii
import base64
from hashlib import sha256 #用于哈希计算

class Wallet:
    """
    钱包类的定义
    """
    def __init__(self):
        """
        钱包初始化时基于椭圆曲线生成一个唯一的秘钥对，代表区块链上一个唯一的账户
        """
        self._private_key = SigningKey.generate(curve=SECP256k1)
        self._public_key = self._private_key.get_verifying_key()

    @property
    def address(self):
        """
        通过公钥生成地址
        """
        h = sha256(self._public_key.to_pem())
        address = base64.b64encode(h.digest())
        return address

    @property
    def publicKey(self):
        """
        返回公钥字符串
        """
        return self._public_key.to_pem()

    def signature(self, message):
        """
        生成消息的数字签名
        """
        h = sha256(message.encode('utf8'))
        signature = binascii.hexlify(self._private_key.sign(h.digest()))
        return signature

    
def verify_signature(publicKey, message, signature):
    """
    验证签名
    """
    verifier = VerifyingKey.from_pem(publicKey)
    h = sha256(message.encode('utf8'))
    return verifier.verify(binascii.unhexlify(signature), h.digest())

'''
if __name__=="__main__":
  wallet= Wallet()
  address=wallet.address
  publicKey=wallet.publicKey
  print("钱包地址:",address)
  print("钱包公钥:",publicKey)
  msg="Testing Blockchain digital wallet."
  sig=wallet.signature(msg)
  print("示例信息签名:",sig)
  if verify_signature(publicKey,msg, sig):
    print('验证通过！')
  else:
    print('验证失败！')
'''