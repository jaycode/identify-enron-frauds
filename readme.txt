In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.
This project attempts to predict the likelihood of someone being a suspect of Enron fraud conspiracy by looking at given dataset. We call the suspects Person of Interest (POI). The dataset contains insider pays to all Enron executives as well as emails sent through their company accounts, and their POI status.
We use machine learning to learn insider pays and emailing habits of POIs and non-POIs and see if we can find a pattern there, then use the model created to predict the likeliness of someone with a particular pattern of being a POI or not.

## Online documentation
- https://jaycode.github.io/enron/identifying-fraud-from-enron-email.html
- https://jaycode.github.io/enron/correlation-analysis.html
- https://jaycode.github.io/enron/bag-of-words-implementation.html

## References
- A Few Useful Things to Know about Machine Learning - Pedro Domingos, Department of Computer Science and Engineering, University of Washington
- Feature Engineering Intro: http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/
- Feature Selection Intro: http://machinelearningmastery.com/an-introduction-to-feature-selection/
- Example of performance improvement by Feature Engineering: [Feature Engineering and Classifier Ensemble for KDD Cup 2010](http://pslcdatashop.org/KDDCup/workshop/papers/kdd2010ntu.pdf)
- Learning about Kernels: https://charlesmartin14.wordpress.com/2012/02/06/kernels_part_1/ and http://www.quora.com/How-does-one-decide-on-which-kernel-to-choose-for-an-SVM-RBF-vs-linear-vs-poly-kernel
- Another good summary of different machine learning techniques and tips: http://blog.bigml.com/2013/02/21/everything-you-wanted-to-know-about-machine-learning-but-were-too-afraid-to-ask-part-two/
- Combining multiple classifiers?: http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
- How to get importance rank in Decision Tree: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
- Evaluation metrix in Machine Learning http://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/
- Good intro on TFIDF: http://www.markhneedham.com/blog/2015/02/15/pythonscikit-learn-calculating-tfidf-on-how-i-met-your-mother-transcripts/
- What is Sparse Matrix? (used in TFIDF): https://en.wikipedia.org/wiki/Sparse_matrix
- Comparison of several ML Algorithms' computational performance: http://ccr.sigcomm.org/online/files/p7-williams.pdf
- Combining Pipelines and Feature Unions: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
- Spot check your algorithms: http://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/
- The best explanation about Precision and Recall: http://rushdishams.blogspot.com/2011/03/precision-and-recall.html
- Understanding AdaBoost algorithm: https://en.wikipedia.org/wiki/AdaBoost
- In depth learning on AdaBoost classifier: https://chrisjmccormick.wordpress.com/2013/12/13/adaboost-tutorial/
- Cross-validation: the illusion of reliable performance estimation - Zolt´an Prekopcs´ak, Tam´as Henk, Csaba G´asp´ar-Papanek
- And tons of other resources.