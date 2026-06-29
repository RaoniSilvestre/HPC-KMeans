# Versão feita em C pura do MPI e OpenMP

## Dependências

- openmp
- openmpi
- clang
- nvidia-hpc-sdk

Se estiver usando nix, pode configurar elas automaticamente com `nix-shell`.

## Uso

Na raiz desse diretório tem um makefile com as seguintes receitas:

| Comando     | Descrição                                                                    |
| ----------- | ---------------------------------------------------------------------------- |
| `all`       | Compila tudo em `out/`. Também é a receita padrão                            |
| `run`       | Compila e executa a versão tradicional                                       |
| `run_gpu`   | Compila e executa a versão com gpu.                                          |
| `run_mpi`   | Compila e executa a versão com `mpi`.                                        |
| `build`     | Compila a versão tradicional.                                                |
| `build_gpu` | Compila a versão com gpu.                                                    |
| `build_mpi` | Compila a versão com `mpi`.                                                  |
| `test`      | Somente testa as receitas compilação do projeto e depois deleta o resultado. |

Por exemplo:

```bash
make run
```

Para configurar o clangd, use o bear para gerar o arquivo
`compile_commands.json`.

```bash
bear -- make -B
```
