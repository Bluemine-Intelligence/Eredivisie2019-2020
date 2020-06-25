---
jupyter:
  jupytext:
    cell_metadata_filter: incorrectly_encoded_metadata,-all
    cell_metadata_json: true
    formats: ipynb,md
    notebook_metadata_filter: title
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: '''Python Interactive'''
    language: python
    name: 174780d9-a10a-430d-8718-268a40edcdc5
  title: Uitslag Eredivisie 2019/2020
  make: >
    conda run jupytext --sync eredivisie2020.md
    conda run jupytext --execute eredivisie2020.md --to notbook
    pandoc -f ipynb+footnotes+implicit_figures eredivisie2020.ipynb --katex -o ./docs/eredivisie2020.html --css ./minimal.css --standalone
---
  
# Uitslag Eredivisie 2019/2020
    
## Het Doel

Recentelijk is er wat ophef onstaan betreffende de beslissingen van de KNVB over de uitslagen van de eredivisie 2019/2020. Om onnodige controverse en verhitte discussies te voorkomen leek het ons een goed idee om de uitslag van de nog ongspeelde wedstrijden bij deze wiskundig, onpartijdig (en niet gehinderd door enige kennis van voetbal) vast te stellen. 

Om dit vast te kunnen stellen is allereerst een statistisch model nodig voor voetbalwedstrijden. Deze kan in principe zo gedetailleerd zijn als we maar willen, toch is het niet praktisch om te beginnen met het simuleren van grassprietjes[^kunstgras], en ook het gedrag van spelers of zelfs eenvoudige spelregels als buitenspel[^buitenspel] blijken toch lastig om precies vast te leggen. Om dit soort problemen uit de weg te gaan is gekozen voor een wat simpeler model waar alleen rekening wordt gehouden met het eindresultaat en niet zo zeer met het proces er aan vooraf.

Lezers die niet geïntereseerd zijn in de wiskundige achtergrond kunnen ook gelijk door gaan naar [de uitslag](#de-uitslag).

[^kunstgras]: Hoe groot het verschil tussen kunstgras en natuurlijk gras ook is.
    
[^buitenspel]: In een poging om toch wat verwarring rondom dit onderwerp weg te nemen volgt hier een elementaire definitie van buitenspel. Laat $V$ het verdedigende team zijn en $h_p(t)$ de horizontale positie van speler $p$ is op tijdstip $t$ en $b(t)$ de positie van de bal, mocht een speler $p$ op de helft van het aan team staan dan staat een speler in buitenspelpositie op tijdstip $t$ indien $$ \int \sum_{v \in V} \theta\bigl(h_v(t) - h_p(t)\bigr)  \delta\bigl(h_v(t) - x\bigr) \, \mathrm{d}x < \tfrac32 \theta\bigl(b(t) - h_p(t)\bigr) $$ Dit geldt mutatis mutandis voor de tegenpartij, tenzij er sprake is van een 'blinde scheids'.

## De Spelregels

Het model dat gebruikt zal worden in dit artikel gaat uit van 2 eenvoudige eigenschappen per team. Namelijk hoe goed ze zijn op aanvallend gebied, en hoe goed hun verdediging is. Kort gezegd gaat dit model er van uit dat de kans dat team A scoort tegen team B evenredig is aan het aanvallend vermogen van team A en het verdedigend vermogen van team B. Verder doet dit model geen aannames over wanneer dit goal dan zou moeten vallen, dus daar gaan het er van uit dat op elk moment in de wedstrijd een goal even waarschijnlijk is.

Het wiskundige model dat hier bij hoort is een Poisson proces met als parameter $a_X d_Y$, waar $a_X$ het aanvallend vermogen van team $X$ is en $d_Y$ het verdedigend vermogen van team $Y$. Dus hoe hoger $d_Y$ des te slechter is team $Y$ in het tegenhouden van goals (andersom was misschien logischer maar dit maakt de analyse wat eenvoudiger).

De kans dat team $X$ dan $k$ goals scoort tegen team $Y$ is:

$$
P(k) = \frac{ (a_X d_Y)^k e^{-a_X d_Y} } { k! }
$$

Maar natuurlijk is het voorspellen van de uitslag van een wedstrijd maar 1 aspect, het is interessanter om de parameters te achterhalen op basis van al gespeelde wedstrijden. Dit kan door middel van Bayesiaanse statistiek. Als we namelijk een reeks wedstrijden hebben van team $X_i$ tegen team $Y_i$, met als uitslag $t_i$ goals tegen $u_i$, dan is de zogenaamde likelihood:

$$
P(t,u | a,d) = \prod_i 
    \frac{ (a_{X_i} d_{Y_i})^{t_i} e^{-a_{X_i} d_{Y_i}} } { t_i! } 
    \frac{ (d_{X_i} a_{Y_i})^{u_i} e^{-d_{X_i} a_{Y_i}} } { u_i! }
$$

voor de analyse is het (zoals vaker) wat eenvoudiger om naar de log-likelihood kijken:

$$
\log(P(t,u | a,d)) 
= \sum_i 
      t_i \log(a_{X_i} d_{Y_i}) - a_{X_i} d_{Y_i} - \log(t_i!)\\ \hphantom{\log(P(t,u | a,d)) = \sum_i }
   +  u_i \log(d_{X_i} a_{Y_i}) - d_{X_i} a_{Y_i} - \log(u_i!)\\
$$

Als we nu de parameters $a$ en $d$ willen achterhalen dan kunnen we gebruik maken van de stelling van Bayes, die zegt dat:

$$
P(a,d | t,u) = \frac{P(t,u|a,d) P(a,d)} {P(t,u)}
$$

als we de prior $P(a,d)$ nog even achterwegen laten dan zien we dat de kansverdeling voor $a$ en $d$ dus evenredig is aan die van $t$ en $u$. Op normalisatie na hebben we dus:

$$
\log(P(a,d | t,u)) = C + \sum_i t_i \log(a_{X_i} d_{Y_i}) + u_i \log(d_{X_i} a_{Y_i}) - a_{X_i} d_{Y_i} - d_{X_i} a_{Y_i}
$$

hiermee kunnen we onder andere bepalen wat de meest waarschijnlijke waarden voor $a$ en $d$ zijn. In dit geval kunnen we simpelweg het maximum vinden door het kritieke punt te bepalen:

$$
\begin{aligned}
0 &= \frac{\partial}{\partial a_Z} \log(P(a,d | t,u))\\
  &= \underbrace{\left(\sum_{X_i = Z} t_i \frac{1}{a_Z} - d_{Y_i}\right)}_{\textrm{thuis}}
    +\underbrace{\left(\sum_{Y_i = Z} u_i \frac{1}{a_Z} - d_{X_i}\right)}_{\textrm{uit}}
\end{aligned}
$$

en dus:

$$
\begin{aligned}
0 &= \biggl(\sum_{X_i = Z} t_i - a_Z d_{Y_i} \biggr)
    +\biggl(\sum_{Y_i = Z} u_i - a_Z d_{X_i} \biggr)\\
  &= \biggl(  \underbrace{\sum_{X_i = Z} t_i }_\textrm{doelpunten thuis} 
           + \underbrace{\sum_{Y_i = Z} u_i }_\textrm{doelpunten uit}\biggr)
    - a_Z \underbrace{\biggl( \sum_{X_i = Z} d_{Y_i}
                            + \sum_{Y_i = Z} d_{X_i} \biggr)}_\textrm{verdedigend vermogen tegenstanders}
\end{aligned}
$$

We kunnen hieruit dus opmaken dat de beste schatting voor de het aanvallend vermogen van een team het totaal aantal doelpunten gedeeld door het verdedigend vermogen van de tegenstander is. Dit zorgt er voor dat doelpunten tegen een slechtere tegenstander de score minder verhogen dan doelpunten tegen een tegenstander die juist erg goed kan verdedigen.

Iets vergelijkbaars geldt voor het verdedigend vermogen:

$$
0 = \biggl(  \underbrace{\sum_{Y_i = Z} t_i }_\textrm{tegendoelpunten thuis} 
           + \underbrace{\sum_{X_i = Z} u_i }_\textrm{tegendoelpunten uit}\biggr) 
    - d_Z \underbrace{\biggl( \sum_{Y_i = Z} a_{Y_i} + \sum_{X_i = Z} a_{X_i} \biggr)}_\textrm{aanvallend vermogen tegenstanders}
$$

Alleen hebben we nu twee vergelijkingen die onderling afhankelijk zijn, en om het echte maximum te vinden moeten we beide tegelijkertijd oplossen. Gelukkig is hier een eenvoudige methode voor, we kunnen namelijk eerst $a$ optimaliseren, daarna $d$ en dit herhalen tot de twee convergeren. De uiteindelijke waarde is dan het echte maximum.

Echter missen we hier nog een subtiel punt. Een gevolg van dit model is namelijk dat de resultaten hetzelfde blijven als we $a$ verdubbelen en $d$ halveren. In andere woorden als er twee keer zo veel wordt geschoten maar er gaan ook maar half zoveel ballen door de verdediging dan is het uiteindelijke resultaat hetzelfde. Dit is op zich niet zo'n probleem maar om te garanderen dat het process convergeert is het handiger om een uniek maximum te hebben. Het simpelst is om te zorgen dat het gemiddelde verdedigend vermogen gelijk is aan 1. Dan is het aanvallend vermogen ongeveer evenredig aan het aantal doelpunten dat het team scoret per wedstrijd.

Een andere mogelijkheid om bovenstaande op te lossen is door een goede prior in te vullen ($P(a,d)$, zie hierboven), maar omdat dit hoe dan ook een arbitraire keuze is en het in dit geval juist beter is om zo min mogelijk subjectieve keuzes te maken, is hier gekozen om de 'impropere' prior $P(a,d)=1$ te gebruiken.
    
## De Uitslag

Nu we een model gekozen hebben en een methode hebben om de parameters te benaderen resteert ons alleen nog om dit los te laten op de resultaten van de eredivisie 2019/2020. De eerste stap is om de model parameters te achterhalen. Dit geeft gelijk ook een overzicht van de onderlinge verhoudingen binnen de eredivisie. 

```python
import pandas as pd
import numpy as np

t = pd.read_csv('goals_thuis.csv',index_col=0).to_numpy()
u = pd.read_csv('goals_uit.csv',  index_col=0).to_numpy()
m = pd.read_csv('aantal_matches.csv', index_col=0).to_numpy()
teams = pd.read_csv('teams.csv', index_col=0)
index = teams.index.rename(None)

N = 18
a = np.full(N,1.0)
d = np.full(N,1.0)

for _ in range(100):
    a = (t.sum(axis=1) + u.sum(axis=0)) / ((m * d).sum(axis=1) + (m * d).sum(axis=0))
    d = (u.sum(axis=1) + t.sum(axis=0)) / ((m * a).sum(axis=1) + (m * a).sum(axis=0))

    a *= d.mean()
    d /= d.mean()

pd.DataFrame(index=index, data={"Aanval" : a, "Verdediging": 1/d}).style.set_precision(2)
```

Wat hier met name opvalt is het enorme offensieve vermogen van Ajax, met op korte afstand AZ en PSV en daarna pas de rest. Ook interessant is dat aanval en verdediging toch vaak samen lijken te gaan, wat de theorie ondersteund dat een goede aanval de beste verdediging is. En zo op het eerste gezicht liggen FC Utrecht en Feyenoord zeer dicht tegen elkaar, wat toch tot een zeer spannende bekerfinale had moeten leiden. 

Nu we de parameters van het model hebben is het vrij eenvoudig om voor de nog ongespeelde wedstrijden de meest waarschijlijke uitslag in te vullen:

```python
# Bereken indices
i,j = np.indices((N,N))
i,j = np.array((i,j))[:, i!=j].reshape(2,-1) # Teams spelen niet tegen zichzelf

# Bekende en voorspelde resultaten
bekend    = (t[i,j], u[i,j])
voorspeld = (np.floor(a[i] * d[j]), np.floor(d[i] * a[j])) # Maximum van Poisson(a_i d_j), Poisson(d_i a_j)

# Genereer dataframe
df = pd.DataFrame(
    {'Thuis':index[i], 
     'Uit':  index[j],
     'Gespeeld': m[i,j] > 0})
df['Voor']  = np.where(df.Gespeeld, bekend[0], voorspeld[0])
df['Tegen'] = np.where(df.Gespeeld, bekend[1], voorspeld[1])
```
waaruit we dan een overzicht kunnen genereren van de gehele competitie inclusief de voorspelde uitslagen:

```python
# Presenteer resultaten
def print_result(voor,tegen,gespeeld):
    if not gespeeld:
        return f"({voor:.0f} - {tegen:.0f})"
    else:
        return f"{voor:.0f} - {tegen:.0f}"

df\
.assign(Uitslag=np.vectorize(print_result)(df.Voor, df.Tegen, df.Gespeeld))\
.pivot(index='Thuis', columns='Uit', values='Uitslag')\
.fillna(' ')\
.rename(columns=teams.TLA)\
.style.set_properties(**{'text-align': 'center'})
```
Maar uiteraard willen we dan ook weten wie er heeft gewonnen, dus zullen we ook het aantal punten en het doelsaldo moeten berekenen:

```python
# Bereken uitslag
uitslag = pd.DataFrame({
    'Gewonnen' : (df.Voor >  df.Tegen).groupby(by=df.Thuis).sum() + (df.Voor <  df.Tegen).groupby(by=df.Uit).sum(),
    'Gelijk'   : (df.Voor == df.Tegen).groupby(by=df.Thuis).sum() + (df.Voor == df.Tegen).groupby(by=df.Uit).sum(),
    'Verloren' : (df.Voor <  df.Tegen).groupby(by=df.Thuis).sum() + (df.Voor >  df.Tegen).groupby(by=df.Uit).sum()})
uitslag['Punten'] = 3 * uitslag.Gewonnen + 1 * uitslag.Gelijk
uitslag['Doelsaldo'] = (df.Voor - df.Tegen).groupby(by=df.Thuis).sum() - (df.Voor - df.Tegen).groupby(by=df.Uit).sum()

# Toon resultaat
uitslag.sort_values(by=['Punten', 'Doelsaldo'], ascending=False).rename_axis(None).style.set_precision(0)
```

En dus is de winaar van de eredivise 2019/2020 Ajax die op doelsaldo nog net weet te winnen van AZ, met uiteindelijk evenveel gewonnen en verloren wedstrijden voor beide. Verder valt FC Utrecht net buiten de top 5 en degraderen ADO en RKC. 

Maar, uiteraard valt er nog wel het een en ander op te merken over deze methodiek, zo wordt er nog geen rekening gehouden met uit/thuis of kunstgras of zelfs veranderingen in de teams over het seizoen heen. Ook is hier alleen *per wedstrijd* de meest waarschijnlijke uitslag berekend, dit is iets heel anders dan een typische uitslag (waarbij het toch te verwachten was dat er een aantal tegendoelpunten zouden zijn geweest tegen Ajax en AZ). En ook als we de vraag willen stellen wie er *waarschijnlijk* had gewonnen is het eigenlijk niet genoeg om simpelweg per wedstrijd de meest waarschijnlijke uitslag te bepalen.

Ter illustratie kunnen we namelijk ook de *verwachte* punten en doelsaldo berekenen:

```python
from scipy.stats import poisson

@np.vectorize
def P(i,j):
    n = 100
    k = np.arange(n)
    p = poisson(a[i] * d[j]).pmf(k)[:,None] * poisson(d[i] * a[j]).pmf(k)[None,:]
    return a[i] * d[j], d[i] * a[j], p[np.tril_indices(n,-1)].sum(), np.diag(p).sum(), p[np.triu_indices(n,1)].sum()

# (her)bereken uitslag
P = df.assign(**dict(zip(['Voor', 'Tegen', 'Gewonnen', 'Gelijk', 'Verloren'], 
    np.where(df.Gespeeld, (df.Voor, df.Tegen, df.Voor >  df.Tegen, df.Voor == df.Tegen, df.Voor < df.Tegen), P(i,j)))))
uitslag = pd.DataFrame({
    'Gewonnen' : P.Gewonnen.groupby(by=P.Thuis).sum() + P.Verloren.groupby(by=P.Uit).sum(),
    'Gelijk'   : P.Gelijk  .groupby(by=P.Thuis).sum() + P.Gelijk  .groupby(by=P.Uit).sum(),
    'Verloren' : P.Verloren.groupby(by=P.Thuis).sum() + P.Gewonnen.groupby(by=P.Uit).sum()})
uitslag['Punten'] = 3 * uitslag.Gewonnen + 1 * uitslag.Gelijk
uitslag['Doelsaldo'] = (P.Voor - P.Tegen).groupby(by=df.Thuis).sum() - (P.Voor - P.Tegen).groupby(by=df.Uit).sum()

# Toon resultaat
uitslag.sort_values(by=['Punten', 'Doelsaldo'], ascending=False).rename_axis(None).style.set_precision(2)
```

We zien hier inderdaad dat Ajax en AFC toch respectievelijk gemiddeld 0.71 en 2.04 wedstrijden zouden verliezen. Waardoor zowel het verwachte aantal punten en doelsaldo voor Ajaz overtuigend hoger is. Merk ook op dat Heracles en Heerenveen nu van positie zijn verwisseld, wat aantoont dat er toch echt een wezenlijk verschil is tussen de twee methoden. In zekere zin lijkt het eerlijker om uit te gaan van gemiddelden dan 1 enkele uitslag per wedstrijd (ook al is het de meest waarschijnlijke uitslag). 

Toch is het niet helemaal terecht om op basis van de gemiddelde te concluderen wie er waarschijnlijk had gewonnen[^dice], om dat te concluderen zullen we toch echte de kans moeten uitrekenen dat Ajax had gewonnen, en een hoger gemiddelde bied geen garantie dat Ajax een grotere kans heeft om hoger te eindigen. Dit is het makkelijkst te bepalden door simpelweg de rest van de competitie te simuleren (een zogeheten monte carlo algoritme), en op basis daarvan te bepalen hoe waarschijnlijk elke positie is per team:

[^dice]: Denk bijvoorbeeld aan 2 dobbelstenen waarbij op de eerste de normale 1 t/m 6 staat en de tweede vijf keer een 1 heeft staan en 1 keer een 100. De tweede heeft een hoger gemiddelde maar als je ze beide rolt rolt de eerste meestal hoger.

```python
volgorde = uitslag.sort_values(by=['Punten', 'Doelsaldo'], ascending=False).index # Onthoud (logische) volgorde
dt,du = poisson(a[i] * d[j]), poisson(d[i] * a[j]) # Kansverdeling voor aantal doelpunten

# N.B. De volgende code is efficient nog snel, dus de berekening kan even duren, verlaag n voor een sneller resultaat
n = 1000
positie = pd.DataFrame(0, index=teams.index, columns=np.arange(1,N+1)).stack()
for _ in range(n):
    # (her)bereken uitslag
    R = df.assign(**dict(zip(['Voor', 'Tegen'],
                             np.where(df.Gespeeld, (df.Voor, df.Tegen), (dt.rvs(), du.rvs())))))
    # Bereken uitslag
    uitslag = pd.DataFrame({
        'Gewonnen' : (R.Voor >  R.Tegen).groupby(by=R.Thuis).sum() + (R.Voor <  R.Tegen).groupby(by=R.Uit).sum(),
        'Gelijk'   : (R.Voor == R.Tegen).groupby(by=R.Thuis).sum() + (R.Voor == R.Tegen).groupby(by=R.Uit).sum(),
        'Verloren' : (R.Voor <  R.Tegen).groupby(by=R.Thuis).sum() + (R.Voor >  R.Tegen).groupby(by=R.Uit).sum()})
    uitslag['Punten'] = 3 * uitslag.Gewonnen + 1 * uitslag.Gelijk
    uitslag['Doelsaldo'] = (R.Voor - R.Tegen).groupby(by=df.Thuis).sum() - (R.Voor - R.Tegen).groupby(by=df.Uit).sum()
    positie[list(zip(uitslag.sort_values(by=['Punten', 'Doelsaldo'], ascending=False).index, np.arange(1,N+1)))] += 1
    
positie = positie*(100/n)
positie = positie.unstack().loc[volgorde]
positie.rename_axis(None).style.background_gradient(cmap='Blues').format("{:.1f}%")
```

Zo op het eertse gezicht lijken de maxima op dezelfde te plek liggen als de uitslag op basis van de gemiddelden, maar merk wel op dat bij veel teams het maximum onder de 50% ligt, wat aangeeft dat ze waarschijnlijker *niet* op die positie zullen eindigen dan wel. En nu is ook duidelijk dat de middenmoot met FC Groningen, Heracles en SC Heerenveen wel erg dicht tegen elkaar aan ligt, wat verklaart waarom de twee voorgaande methodes verschillende resultaten gaven voor die clubs. 

Nadeel van deze methode is dat het moeilijk is om de teams te rangschikken, waardoor er nu dus aardig wat discussie mogelijk is over de uitslag, zo *had* Utrecht bijvoorbeeld best nog 5e kunnen worden, en de degradatie van ADO Den Haag staat ook nog niet volledig vast. Dit zou als argument gebruikt kunnen worden dat het *terecht* was om de competitie nietig te verklaren. Toch zal het voor fans van FC Utrecht wrang blijven dat hen een reële kans op de Europa League is ontzegt.

En dus resteert nu alsnog de vraag wat eerlijk is. Kijken we naar de meest waarschijnlijke uitslag per wedstrijd, of het verwachte aantal punten, of de kans die elk team maakte op een bepaalde positie te eindigen, of genereren we gewoon een willekeurige uitslag (als een soort loting) en baseren we alles daarop?
