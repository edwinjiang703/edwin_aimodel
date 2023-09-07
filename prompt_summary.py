from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000,openai_api_key=OPENAI_API_KEY)

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system","你将获得关于同一主题的{num}篇文章（用-----------标签分隔）,首先总结每篇文章的论点。然后指出哪篇文章提出了更好的论点，并解释原因."),
        ("human","{user_input}"),
    ]
)

# summary_template = ChatPromptTemplate.from_messages(
#     [
#         ("system","有{num}文章"),
#         ("human","{user_input}"),
#     ]
# )


messages=summary_template.format_messages(
    num=2,
    user_input='''1.认为“道可道”中的第一个“道”，指的是道理，如仁义礼智之类；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，所谓“道可道，非常道”，指的是可以言说的道理，不是恒久存在的“道”，恒久存在的“道”不可言说。如苏辙说：“莫非道也。而可道者不可常，惟不可道，而后可常耳。今夫仁义礼智，此道之可道者也。然而仁不可以为义，而礼不可以为智，可道之不可常如此。……而道常不变，不可道之能常如此。”蒋锡昌说：“此道为世人所习称之道，即今人所谓‘道理’也，第一‘道’字应从是解。《广雅·释诂》二：‘道，说也’，第二‘道’字应从是解。‘常’乃真常不易之义，在文法上为区别词。……第三‘道’字即二十五章‘道法自然’之‘道’，……乃老子学说之总名也”。陈鼓应说：“第一个‘道’字是人们习称之道，即今人所谓‘道理’。第二个‘道’字，是指言说的意思。第三个‘道’字，是老子哲学上的专有名词，在本章它意指构成宇宙的实体与动力。……‘常道’之‘常’，为真常、永恒之意。……可以用言词表达的道，就不是常道”。
-----------
2.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，“道可道，非常道”，指可以言说的“道”，就不是恒久存在的“道”。如张默生说：“‘道’，指宇宙的本体而言。……‘常’，是经常不变的意思。……可以说出来的道，便不是经常不变的道”。董平说：“第一个‘道’字与‘可道’之‘道’，内涵并不相同。第一个‘道’字，是老子所揭示的作为宇宙本根之‘道’；‘可道’之‘道’，则是‘言说’的意思。……这里的大意就是说：凡一切可以言说之‘道’，都不是‘常道’或永恒之‘道’”。汤漳平等说：“第一句中的三个‘道’，第一、三均指形上之‘道’，中间的‘道’作动词，为可言之义。……道可知而可行，但非恒久不变之道”。
--------
3.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，则指的是平常人所讲之道、常俗之道。因此，“道可道，非常道”，指“道”是可以言说的，但它不是平常人所谓的道或常俗之道。如李荣说：“道者，虚极之理也。夫论虚极之理，不可以有无分其象，不可以上下格其真。……圣人欲坦兹玄路，开以教门，借圆通之名，目虚极之理，以理可名，称之可道。故曰‘吾不知其名，字之曰道’。非常道者，非是人间常俗之道也。人间常俗之道，贵之以礼义，尚之以浮华，丧身以成名，忘己而徇利。”司马光说：“世俗之谈道者，皆曰道体微妙，不可名言。老子以为不然，曰道亦可言道耳，然非常人之所谓道也。……常人之所谓道者，凝滞于物。”裘锡圭说：“到目前为止，可以说，几乎从战国开始，大家都把‘可道’之‘道’……看成老子所否定的，把‘常道’‘常名’看成老子所肯定的。这种看法其实有它不合理的地方，……‘道’是可以说的。《老子》这个《道经》第一章，开宗明义是要讲他的‘道’。第一个‘道’字，理所应当，也是讲他要讲的‘道’：道是可以言说的。……那么这个‘恒’字应该怎么讲？我认为很简单，‘恒’字在古代作定语用，经常是‘平常’‘恒常’的意思。……‘道’是可以言说的，但是我要讲的这个‘道’，不是‘恒道’，它不是一般人所讲的‘道’。
'''
)

chat_result = chat_model(messages)
print(chat_result.content)