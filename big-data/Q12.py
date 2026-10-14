from Bio import Entrez
Entrez.email = "ahdezneytiry@gmail.com" 
gene_ids = ['7157', '1956', '348', '7124', '7422', '7040', '4524', '2064', '3586', '6774']
for gene_id in gene_ids:
    handle = Entrez.efetch(db="gene", id=gene_id, rettype="fasta", retmode="text")
    fasta_data = handle.read()
    handle.close()
    with open(f"Aspirin_gene_{gene_id}.fasta", "w", encoding="utf-8") as f:
        f.write(fasta_data)
    print(f"已保存：Aspirin_gene_{gene_id}.fasta")