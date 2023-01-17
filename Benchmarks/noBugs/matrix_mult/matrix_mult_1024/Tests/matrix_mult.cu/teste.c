#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    int size = atoi(argv[1]);
    size *= size * 2;

    int vetor[size];

    for(int i = 0; i < size; i++){
        scanf("%d", &vetor[i]);
    }

    printf("imprimir\n");
    for(int i = 0; i < size; i++){
        printf("%d ", vetor[i]);
    }
    printf("\n");
    return 0;
}
