Bayes
=====

Naive Bayes classifier, C-version (compile with -lm).

See http://en.wikipedia.org/wiki/Naive_Bayes_classifier

```
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

```

