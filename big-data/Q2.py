from Bio import Entrez
Entrez.email = "ahdezneytiry@gmail.com"
h = Entrez.esearch(db="nucleotide", term="COI[Gene]", retmax=10)
ids = Entrez.read(h)["IdList"]
h.close()
for i in ids:
	f = Entrez.efetch(db="nucleotide", id=i, rettype="fasta", retmode="text")
	d = f.read()
	f.close()
	with open(f"COI_{i}.fasta", "w", encoding="utf-8") as w:
		w.write(d)
