Introduction: 
Our project is about Twitter sentiment analysis, in which we collected data from Twitter using a Python scraper. Once we obtained the data, we pre-processed it because the data contains hashtags and different elements such as stop words, extra spaces, URLs, etc. After pre-processing, we used different algorithms to treat our pre-processed data and wrote functions for sentiment analysis. This helped us to perform sentiment analysis.

Algorithm Used In Sentiment Analysis:
1.	NLTK (Natural Language Toolkit)
2.	STANZA
3.	TextBlob
4.	CoreNlp
We performed these algorithms on our data set separately and evaluated which one gave the best results. After evaluation, we decided to use NLTK and developed a prediction system using it. This allows the user to input any data and get a result.
Emotions In Our Project: 
Our model can detect 5 emotions:
1.	Happy (meaning the user is happy)
2.	Joyful (meaning the user is very happy)
3.	Angry (meaning the user is very upset or angry)
4.	Sad (meaning the user is sad or upset)
5.	Neutral (meaning the emotion is neither good nor bad)

How To Run Predictive System: 


I have defined the steps below to help you run the predictive system step by step:
1.	You have been provided with a file named "Testing of Sentiment". To open this file, you need software called Jupyter Notebook. To install it, go to your command prompt and type "pip install jupyter". It will take some time to install. Once it's complete, type "jupyter notebook" on the command prompt. It will open in any browser, and you can run the file on it.
2.	You will see an upload option. Upload the file that was sent to you named "Testing of Sentiment".
3.	You will also receive a CSV file named "Final.csv". Upload it in the same way.
4.	Open the "Testing of Sentiment" file that you uploaded by double-clicking it.
5.	You will see some cells that need to be run one by one, such as "pip install pandas", etc. To run these cells, press Shift+Enter on your keyboard. You need to do this for each cell.
6.	Go to cell number 6 and check if the second file you uploaded named "Final.csv" is mentioned here. Confirm it and press Shift+Enter.
7.	In the last cell, input the data for which you want to check the sentiment, and the result will be shown at the end of the cell.


