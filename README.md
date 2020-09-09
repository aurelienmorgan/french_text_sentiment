# French-Text Sentiment Analysis


**Welcome to this project&nbsp;!**

<p style="text-align: justify;">
The topic covered here is Sentiment Analysis in texts written
<span style="border-bottom:1px solid #000; text-decoration:underline; display:inline-block;">
in French language</span>.
For that, we employ a Recurrent Neural Network that we build and run thru the Tensorflow / Keras framework.
<br />
The architecture of the model is based on dual bi-directionnal GRU cells
and it employs fastText word embeddings.
We train this model using tranfer learning from rated product reviews
that have been  web-scrapped using the BeautifulSoup python library
<font style="color: darkgray;"><em>(the web-scraping code is not provided,
but the collected data is)</em></font>.
<br />
<br />
<image style="margin: 0px 10px 0px 0px;"  align="left" src="./images/project_structure.png?uncahce=1234" />The figure on the left shows the structure of this project.
There are two key points to notice&nbsp;:
<ul style="line-height:22px">
<li style="margin: 0px 0px 0px 300px;">A dedicated custom python package named <b>my_NLP_RNN_fr_lib</b> has been developped to serve this project.</li>
<li style="margin: 0px 0px 0px 300px;">There's a whole sub-section to the herein project, detailled separately, on <b>hyperparameters optimization</b>,.
It can be found <a href="https://github.com/aurelienmorgan/french_text_sentiment/blob/master/model/optimization/"
target="_blank">there&nbsp;<img
href='.' src='./images/target_blank.png' style='vertical-align: baseline; display: inline;' /></a>.
Spoiler alert&nbsp;: we deal with random search first, then XGBoost + scikit-learn
are called to get an extra edge.
</li>
</ul>

The French-Text Sentiment Analysis project we're dealing with here is explained in details and accompagnied with full running python code
in a walkthrough Jupyter Notebook.

</p>






<div style="width: 100%;">
    <center>
        <div>
            <a href="https://htmlpreview.github.io/?https://github.com/aurelienmorgan/french_text_sentiment/blob/master/main.html?uncache=654645"
                target="self"><img alt="Jupyter Notebook" src="./images/jupyter_notebook.png?uncache=1234" height="40px" /></a>
        </div>
    </center>
</div>
<br />
<br />




KEYWORDS :
	```Tensorflow```, ```Keras```,
	```GRU```, ```RNN```, ```NLP```, ```fastText```,
	```web-scraping```, ```BeautifulSoup```, 
	```transfer learning```, ```french sentiment analysis```