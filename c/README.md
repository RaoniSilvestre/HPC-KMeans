# Versão feita em C pura do MPI e OpenMP

## Dependências

Na raiz desse diretório tem um makefile com as seguintes receitas:

| Comando     | Descrição                                                                    |
| ----------- | ---------------------------------------------------------------------------- |
| `all`       | Compila tudo em `out/`. Também é a receita padrão                            |
| `run`       | Compila e executa a versão sem `mpi`                                         |
| `run_mpi`   | Compila e executa a versão com `mpi`.                                        |
| `build`     | Compila a versão sem `mpi`.                                                  |
| `build_mpi` | Compila a versão com `mpi`.                                                  |
| `test`      | Somente testa as receitas compilação do projeto e depois deleta o resultado. |

Por exemplo:

```bash
make run
```
