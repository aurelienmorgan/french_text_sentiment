








**Welcome to this <u>Deep Learning Hyperparameters Optimization</u> section&nbsp;!**

<p style="text-align: justify;">
In here, we optimize the hyperparameters of the Tensorflow/Keras Reccurent Neural Network
that we use in <a href="https://github.com/aurelienmorgan/french_text_sentiment"
target="_blank">that other (parent) project
<img href='.' src='../../images/target_blank.png' style='vertical-align: baseline; display: inline;' /></a>.
<br />
The topic covered there is Sentiment Analysis in texts written
<span style="border-bottom:1px solid #000; text-decoration:underline; display:inline-block;">in French language</span>.
The architecture of the model is based on dual bi-directionnal GRU cells,
it employs fastText word embeddings and we train it using tranfer learning from web-scrapped
rated product reviews. <em>You should go check that out&nbsp;!</em>
</p>

<p style="text-align: justify;">
To take care of the hyperparameters optimization,
we operate the procedure depicted in the below process chart on a small subset
of the training data&nbsp;:
</p>
<center>
<b>NLP hyperparameters random search on (XGBoost) steroïds</b>
<a href="..\..\images\hyperparameters.png" target="_blank"><img src="..\..\images\hyperparameters.png" style="width: 100%;"/></a>
<em><small>click to enlarge</small></em>
</center>
<br />

It is explained in details and accompagnied with full running python code
in a dedicated walkthrough Jupyter Notebook.

</p>


<div style="width: 100%;">
    <center>
        <div>
            <a href="https://htmlpreview.github.io/?https://github.com/aurelienmorgan/french_text_sentiment/blob/master/model/optimization/hyperparameters_values_search.html"
                target="self"><img alt="Jupyter Notebook" src="../../images/jupyter_notebook.png?uncache=1234" height="40px" /></a>
        </div>
    </center>
</div>
<br />
<br />

KEYWORDS :
	```Tensorflow```, ```Keras```,
	```hyperparameters optimization```, 
	```Dask```, ```multiprocessing```, 
	```XGBoost```, ```Scikit-Learn```





