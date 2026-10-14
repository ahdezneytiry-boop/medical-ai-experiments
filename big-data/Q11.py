from Bio import Entrez
Entrez.email = "ahdezneytiry@gmail.com" 
handle = Entrez.esearch(db="gene", term="Aspirin", retmax=10)
record = Entrez.read(handle)
handle.close()
print(record["IdList"])

