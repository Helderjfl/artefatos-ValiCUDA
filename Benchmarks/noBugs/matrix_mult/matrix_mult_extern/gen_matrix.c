#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const *argv[])
{
    int N = atoi(argv[1]);
    FILE *fp;
    char arquivo[50] = "input";

    strcat(arquivo, argv[1]);
    strcat(arquivo, ".txt");

    fp = fopen(arquivo, "w");

    int cont = 0;

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            fprintf(fp, "%d ", i);
        fprintf(fp, "\n");
    }

    cont = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            fprintf(fp, "%d ", i);
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}
