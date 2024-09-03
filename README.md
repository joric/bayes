# Bayes

Naive Bayes classifier, C-version (compile with -lm).

See http://en.wikipedia.org/wiki/Naive_Bayes_classifier

Let C is a certain class, F is a certain feature. All we have to do is to find max p(C|F1,..,Fn).

Accoring to the Bayes theorem: p(C|F1,...,Fn) = p(C) * p(F1,...,Fn|C) / p(F1,...,Fn)

Using the joint probabilty: p(F1,...,Fn|C) = p(C,F1,..,Fn)

With indepence assumptions: p(C,F1,..,Fn) = p(F1|C) * ... * p(Fn|C)

Finally: p(C|F1,..,Fn) = p(C) * p(F1|C) * ... * p(Fn|C) / (p(F1) * ... * p(Fn))

I.e. posterior = prior * likehood / evidence, where:

* posterior = p(C|F1,..,Fn), the result we need
* prior = p(C), constant to C, generally 1 / number of classes
* evidence = p(F1,..,Fn) = p(F1) * ... * p(Fn), constant to F.
* likehood = p(F1|C) * ... * p(Fn|C)

If posterior hits a certain margin, e.g. 0.99, we consider that questioning is done.
Otherwise we pick features that decrease entropy in all possible cases,
both for YES and NO and everything in between.
To determine the best question we have to calculate results for all possible outcomes
and calculate entropy sum across all classes, which is:

entropy = -posterior * log(posterior)

That's, basically, all. Mind that p(C,F1,..,Fn) may exceed 1 on a redundant set of features.


## References

* https://github.com/ttezel/bayes A Naive-Bayes classifier for node.js
* https://youtu.be/HZGCoVF3YvM Bayes theorem, the geometry of changing beliefs
