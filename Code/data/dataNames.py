from Bio import SeqIO

record = SeqIO.parse("smallData.fa", "fasta")

text = open('text.txt', 'w')

for rec in record:
    for c in rec.name:
        text.write(c)
    text.write('\n')
