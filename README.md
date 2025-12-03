# Heterolattices
This repository contains code and data for the heterolattices discovery. The code including structure generation through Zur algorithm and interlayer distance optimization, elastic energy estimation for further screening, Fermi level alignment and band crossing evaluation for interlayer coupling analysis. Lists of commensurate pairs of monolayers originated from monolayers in C2DB and Jarvis databases are contained.

<img width="4076" height="2204" alt="PipelineHeterolattices" src="https://github.com/user-attachments/assets/7773bbea-ad8f-4176-bc1a-b3fdda91c90b" />


### Repository Structure

1. results/: lists of pairs of monolayers that can be constructed as commensurate heterolattices.
   
2. core/: scritps for Zur algorithm, elastic energy estimation, Fermi level alignment and band crossing evaluation.
   
3. Jarvis/: notebook files work with Jarvis database. Include Analysis of band crossings and structure generation.
   
4. C2DB/: notebook file work with C2DB database on band crossing analysis. Python file of structure generation. To generate, lists of structures in results/ can be used.

```python hetero_gen.py```
