/* (c) joric^proxium 2011, public domain

Naive Bayes Classifier (compile with -lm)

http://en.wikipedia.org/wiki/Naive_Bayes_classifier

In short, let C is a certain class, F is a certain feature.
All we have to do is to find max p(C|F1,..,Fn).

Accoring to the Bayes theorem:

p(C|F1,...,Fn) = p(C) * p(F1,...,Fn|C) / p(F1,...,Fn) 

Using the joint probabilty, p(F1,...,Fn|C) = p(C,F1,..,Fn)
With indepence assumptions: p(C,F1,..,Fn) = p(F1|C) * ... * p(Fn|C)

Finally: p(C|F1,..,Fn) = p(C) * p(F1|C) * ... * p(Fn|C) / (p(F1) * ... * p(Fn))

i.e. [posterior] = [prior] * [likehood] / [evidence], where:

[posterior] = p(C|F1,..,Fn), the result we need
[prior] = p(C), constant to C, generally 1 / number of classes
[evidence] = p(F1,..,Fn) = p(F1) * ... * p(Fn), constant to F.
[likehood] = p(F1|C) * ... * p(Fn|C)

P.S. p(C|F1,...,Fn) may exceed 1 on a redundant set of features.
While querying pick questions that decrease entropy in all cases.

*/

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <math.h>
#include <string.h>

#define BUF_SIZE 256
#define EPS 0.000001
#define INF 10000000
#define MAX_CLASSES 128
#define MAX_FEATURES 128
#define MAX_QUESTIONS 128
#define MAX_ANSWERS (MAX_CLASSES * MAX_QUESTIONS)
#define P(i,n) ((i <= 0 || n <= 0) ? EPS : i / (double) n)
#define RMCHR(s,c) strchr(s,c) ? *strchr(s,c) = 0 : 0
#define TRIM(s) RMCHR(s,'\n'), RMCHR(s,'\r')

int learn = 1;
int debug = 1;

typedef struct {
    char name[BUF_SIZE];
    double p;
} class_t;

typedef struct {
    char text[BUF_SIZE];
    int skip;
    double a;
    double b;
} question_t;

typedef struct {
    int q;
    int value;
} feature_t;

typedef struct {
    int c;
    int q;
    int value;
} answer_t;

int classes = 0;
int answers = 0;
int features = 0;
int questions = 0;

class_t C[MAX_CLASSES];
answer_t A[MAX_ANSWERS];
feature_t F[MAX_FEATURES];
question_t Q[MAX_QUESTIONS];

enum {
    NO = 0,
    YES = 1,
    UNKNOWN = 2,
    PROBABLY = 3,
    DOUBTFUL = 4,
};

int V[] = { NO, YES, UNKNOWN, PROBABLY, DOUBTFUL };

double weights[] = { 0.0, 1.0, 0.5, 0.75, 0.25 };

int values = 5;

const char *FILE_CLASSES = "bayes_classes.txt";
const char *FILE_QUESTIONS = "bayes_questions.txt";
const char *FILE_ANSWERS = "bayes_answers.txt";

const char *TOKEN_A = "%d %d %d\n";
const char *TOKEN_WSTR = "%s\n";

int load()
{
    FILE *fp;
    int i = 0;

    if (fp = fopen(FILE_CLASSES, "r"))
    {
        for (i = 0; i < MAX_QUESTIONS && fgets(C[i].name, BUF_SIZE, fp) > 0; i++)
            TRIM(C[i].name);
        classes = i;
        fclose(fp);
    }

    if (fp = fopen(FILE_QUESTIONS, "r"))
    {
        for (i = 0; i < MAX_QUESTIONS && fgets(Q[i].text, BUF_SIZE, fp) > 0; i++)
            TRIM(Q[i].text);
        questions = i;
        fclose(fp);
    }

    if (fp = fopen(FILE_ANSWERS, "r"))
    {
        for (i = 0; i < MAX_ANSWERS && fscanf(fp, TOKEN_A, &A[i].c, &A[i].q, &A[i].value) > 0; i++);
        answers = i;
        fclose(fp);
    }

    return 0;
}

int save()
{
    FILE *fp;

    int i = 0;

    if (fp = fopen(FILE_CLASSES, "w"))
    {
        for (i = 0; i < classes && fprintf(fp, TOKEN_WSTR, C[i].name) > 0; i++);
        fclose(fp);
    }

    if (fp = fopen(FILE_QUESTIONS, "w"))
    {
        for (i = 0; i < questions && fprintf(fp, TOKEN_WSTR, Q[i].text) > 0; i++);
        fclose(fp);
    }

    if (fp = fopen(FILE_ANSWERS, "w"))
    {
        for (i = 0; i < answers && fprintf(fp, TOKEN_A, A[i].c, A[i].q, A[i].value) > 0; i++);
        fclose(fp);
    }

    return 0;
}

int choose(char *msg)
{
    printf("%s (y/p/u/d/n): ", msg);

    while (1)
    {
        static char buf[BUF_SIZE];
        fgets(buf, BUF_SIZE, stdin);
        switch (buf[0])
        {
            case 'y': return YES;
            case 'n': return NO;
            case 'u': return UNKNOWN;
            case 'p': return PROBABLY;
            case 'd': return DOUBTFUL;
        }
    }
}

char *input(char *msg)
{
    static char buf[BUF_SIZE];
    printf("%s", msg);
    fgets(buf, BUF_SIZE, stdin);
    TRIM(buf);
    return buf;
}

void normalize()
{
    int i;
    double scale;
    double sum = 0;

    for (i = 0; i < classes; i++)
        sum += C[i].p;

    scale = 1.0 / sum;

    for (i = 0; i < classes; i++)
    {
        double p = C[i].p;

        p = (sum < EPS) ? EPS : p * scale;

        if (p < EPS)
            p = EPS;

        C[i].p = p;
    }
}

int get_value(int c, int q)
{
    int i;
    for (i = 0; i < answers; i++)
    {
        if (A[i].c == c && A[i].q == q)
        {
            return A[i].value;
        }
    }

    return UNKNOWN;
}

double correlation(double avg, double p)
{
    double res = 0;
    double k = 0;
    double t = 0.5;

    if (p > t)
    {
        k = (p - t) / (1 - t);
        res = t + (avg - t) * k;
    }
    else
    {
        k = ((1 - p) - t) / (1 - t);
        res = t + ((1 - avg) - t) * k;
    }

    return res;
}

double prob(int a, double sum, int count, int total)
{
    double p = weights[a];

    double avg = count ? sum / (double)count : 0;

    double res = correlation(avg, p);

    return P(res, total);
}

double pf(int q, int a)
{
    double sum = 0;

    int i;

    for (i = 0; i < classes; i++)
    {
        sum += weights[get_value(i, q)];
    }

    return prob(a, sum, classes, classes);
}

double pfc(int q, int a, int c)
{
    double sum = weights[get_value(c, q)];

    return prob(a, sum, 1, questions);
}

void calc_p()
{
    int i, j;

    double prior = 1.0 / classes;

    double evidence = 1.0;

    for (i = 0; i < features; i++)
    {
        evidence *= pf(F[i].q, F[i].value);
    }

    for (j = 0; j < classes; j++)
    {
        double posterior;
        double likehood = 1.0;

        for (i = 0; i < features; i++)
        {
            likehood *= pfc(F[i].q, F[i].value, j);
        }

        posterior = prior * likehood / evidence;

        C[j].p = posterior;
    }

    normalize();
}

double calc_entropy()
{
    int i;
    double e = 0;

    calc_p();

    for (i = 0; i < classes; i++)
    {
        double p = C[i].p;
        e += -p * log(p);
    }

    return e;
}

void add_feature(int q, int value)
{
    F[features].q = q;
    F[features].value = value;
    features++;
}

void remove_feature()
{
    if (features > 0)
        features--;
}

int best_question()
{
    int j, k;

    double entropy = calc_entropy();

    int res = -1;

    for (j = 0; j < questions; j++)
    {
        if (!Q[j].skip)
        {
            double ek_max = -INF;
            double ek_min = INF;

            for (k = 0; k < values; k++)
            {
                if (V[k] == YES || V[k] == NO)
                {
                    double e;

                    add_feature(j, V[k]);

                    e = calc_entropy();

                    remove_feature();

                    if (e > ek_max)
                        ek_max = e;

                    if (e < ek_min)
                        ek_min = e;

                    printf("%lf ", e);
                }
            }

            if (entropy > ek_max)
            {
                entropy = ek_max;
                Q[j].a = ek_min;
                Q[j].b = ek_max;
                res = j;
            }

            printf("%d. %s (%lf) -> %d\n", j, Q[j].text, ek_max, res);
        }
    }

    return res;
}

void reset()
{
    int i, j;

    for (i = 0; i < classes; i++)
        C[i].p = 0;

    for (j = 0; j < questions; j++)
        Q[j].skip = 0;

    features = 0;

    calc_p();
}

int add_class(char *str)
{
    int i, n = -1;

    if (!str || !strlen(str))
        return -1;

    for (i = 0; i < classes; i++)
    {
        if (strcmp(C[i].name, str) == 0)
        {
            printf("Oh, I know this object!\n");
            n = i;
        }
    }

    if (n == -1)
    {
        n = classes;
        strcpy(C[n].name, str);
        classes++;
    }
    return n;
}

int add_question(char *str)
{
    int i, n = -1;

    if (!str || !strlen(str))
        return -1;

    for (i = 0; i < questions; i++)
    {
        if (strcmp(Q[i].text, str) == 0)
        {
            printf("Oh, I know that question!\n");
            n = i;
        }
    }

    if (n == -1)
    {
        n = questions;
        strcpy(Q[n].text, str);
        questions++;
    }

    return n;
}

int add_answer(int c, int q, int value)
{
    int n = -1;
    int i;

    for (i = 0; i < answers; i++)
    {
        if (A[i].c == c && A[i].q == q)
        {
            n = i;
            A[i].value = value;
        }
    }

    if (n == -1)
    {
        n = answers;
        A[n].c = c;
        A[n].q = q;
        A[n].value = value;
        answers++;
    }

    printf("%s - %s - %d\n", C[c].name, Q[A[n].q].text, A[n].value);

    return n;
}

void won(int c)
{
    printf("I won!\n");

    if (learn)
    {
        int i;
        for (i = 0; i < features; i++)
            add_answer(c, F[i].q, F[i].value);

        save();
    }
}

void lost(int k)
{
    printf("I lost!\n");

    if (learn)
    {
        int q;
        int i;
        int c = add_class(input("Name your object (default - skip): "));

        if (c == -1)
            return;

        printf("Add your question (default - none): ");

        q = add_question( input("") );

        if (q != -1)
        {
            int p = choose("And the answer is?");
            add_answer(c, q, p);
        }

        for (i = 0; i < features; i++)
            add_answer(c, F[i].q, F[i].value);

        save();
    }
}

int top_class()
{
    int o = 0;
    double p = 0;
    int i;
    for (i = 0; i < classes; i++)
        if (C[i].p > p)
            p = C[i].p, o = i;
    return o;
}

void dump(int q, double e)
{
    int i, j;

    for (i = 0; i < classes; i++)
    {
        printf("%-21.21s %f ", C[i].name, C[i].p);

        for (j = 0; j < questions; j++)
        {
            int v = get_value(i, j);

            printf("%s%d", j ? " " : "", v);
        }

        printf("\n");
    }

    printf("%-21.21s %f ", "Entropy:", e);

    for (j = 0; j < questions; j++)
    {
        printf("%s%s", j ? " " : "", j == q ? "^" : " ");
    }

    printf("\n");

    if (q >= 0 && q < questions)
    {
        printf("%s %s (-%lf/+%lf)\n", "Best question:", Q[q].text, Q[q].a, Q[q].b);
    }
}

int next_question()
{
    int i;
    for (i = 0; i < questions; i++)
        if (!Q[i].skip)
            return i;
    return 0;
}

int main(int argc, char **argv)
{
    int i, c;

    load();

    if (classes==0)
    {
         int i;

        // example data
        const char * ex_cl[] = {"Arnold Schwarzenegger", "Chuck Norris",
            "Barack Obama", "Harry Potter", "Emma Watson", "Adolf Hitler",
            "Osama Bin Laden", "Lara Croft"};

        const char * ex_q[] = {"Am I a movie actor?","Am I wearing a beard?",
            "Am I born in America?", "Am I real?","Am I a woman?",
            "Am I a terrorist?", "Am I evil?","Am I black?"};

        int ex_a[][3] = {{0,0,1}, {1,0,1}, {2,0,0}, {0,1,0}, {1,1,1},
            {2,1,0}, {0,2,0}, {1,2,1}, {2,2,1}, {3,3,0}, {3,0,0}, {1,3,1},
            {4,4,1}, {4,3,1}, {1,4,0}, {3,4,0}, {5,2,0}, {5,4,0}, {5,3,1},
            {5,1,0}, {5,0,0}, {6,5,1}, {6,1,1}, {7,3,0}, {7,5,0}, {7,4,1},
            {1,5,0}, {0,5,0}, {0,3,1}, {0,4,0}, {2,5,0}, {2,3,1}, {2,4,0},
            {3,5,0}, {3,2,0}, {3,1,0}, {4,5,0}, {4,2,0}, {4,0,1}, {4,1,0},
            {6,3,1}, {6,0,0}, {6,4,0}, {6,2,0}, {7,2,0}, {7,1,0}, {7,0,0},
            {6,6,1}, {5,6,1}, {5,5,0}, {0,6,0}, {1,6,0}, {2,6,0}, {3,6,0},
            {4,6,0}, {7,6,0}, {0,7,0}, {1,7,0}, {2,7,1}, {3,7,0}, {4,7,0},
            {5,7,0}, {6,7,0}, {7,7,0}, {1,7,0}, {1,3,1}, {1,5,0}, {1,6,0},
            {1,4,0}, {1,1,1}, {1,1,1}, {5,0,0}, {5,3,1}, {5,6,1}, {5,1,0},
            {3,0,0}, {3,3,0}, {3,4,0}, {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1},
            {7,0,0}, {7,3,0}, {7,4,1}, {1,1,1}, {0,1,0}, {0,4,0}, {1,1,1},
            {0,1,0}, {0,4,0}, {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1}, {1,1,1},
            {1,1,1}, {3,0,0}, {3,3,0}, {3,4,0}};

        classes = sizeof(ex_cl)/sizeof(*ex_cl);
        for (i=0; i<classes; i++)
            strcpy(C[i].name, ex_cl[i]);

        questions = sizeof(ex_q)/sizeof(*ex_q);
        for (i=0; i<questions; i++)
            strcpy(Q[i].text, ex_q[i]);

        answers = sizeof(ex_a)/sizeof(*ex_a);
        for (i=0; i<answers; i++)
            A[i].c = ex_a[i][0], A[i].q = ex_a[i][1], A[i].value = ex_a[i][2];
    }

    while (1)
    {
        printf("Classes: %d, Questions: %d, Answers: %d\n", classes, questions, answers);

        if (!questions)
            lost(0);

        reset();

        for (i = 0; i < questions; i++)
        {
            int value;
            int q = best_question();

            double e = calc_entropy();
            c = top_class();

            if (debug)
                dump(q < 0 ? i : q, e);

            if (q == -1 && C[c].p > 0.5)
                break;

            if (q == -1)
                q = next_question();

            printf("%d. ", i + 1);

            value = choose(Q[q].text);

            add_feature(q, value);

            Q[q].skip = 1;
        }

        printf("I'm thinking of... %s (%f)", C[c].name, C[c].p);

        choose("") ? won(c) : lost(c);

        printf("---\n");
    }

    return 0;
}
