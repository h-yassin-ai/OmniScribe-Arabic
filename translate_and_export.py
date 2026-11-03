import os
import re

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    entries = []
    blocks = content.strip().split("\n\n")
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) >= 3:
            idx = lines[0].strip()
            timing = lines[1].strip()
            text = " ".join(lines[2:]).strip()
            entries.append({"index": idx, "timing": timing, "text": text})
    return entries

# Traduction soignee et fidele du discours spirituel et theologique
translations_fr = [
    # 1 - 5
    "En ce qui concerne les paroles des gnostiques et des grands maîtres,",
    "certaines âmes et certaines passions déforment les propos tenus à leur sujet,",
    "comme l'Imam al-Haddad le mentionne dans son livre « La Risâlat al-Mu'âwanah ».",
    "Il a rappelé ces paroles, qui sont celles de nos maîtres, de nos pieux prédécesseurs d'Orient et d'Occident,",
    "ceux qu'Allah a manifestés par la lumière, le bien et la piété.",
    # 6 - 10
    "Qu'il s'agisse de Sayyidunâ al-Fâriq, de Sayyidunâ Ibn 'Arabî, ou des maîtres de cette voie,",
    "comme 'Abd al-Ghanî an-Nâbulusî ou ceux qui les ont précédés et suivis,",
    "ces grands maîtres, comme 'Abd al-Karîm al-Jîlî,",
    "il est universellement reconnu qu'ils font partie de l'élite de la communauté de la sagesse,",
    "des meilleurs des vertueux, des savants et des gens de la connaissance d'Allah Tout-Puissant.",
    # 11 - 15
    "Le fait que leurs paroles soient difficiles d'accès pour beaucoup de gens, qui n'en comprennent pas le sens",
    "ni n'en saisissent la finalité, ne diminue en rien leur rang éminent.",
    "De même, la mise en garde de certains gnostiques contre la lecture de leurs livres est juste,",
    "car ils craignent que ces textes soient compris de travers, provoquant égarement et déviation.",
    "C'est ainsi que la majorité de nos prédécesseurs n'aimaient pas que le grand public et les disciples débutants",
    # 16 - 20
    "se plongent dans ces ouvrages, comme les livres d'Ibn 'Arabî et d'autres.",
    "L'Imam al-Haddad disait : si le lecteur dit : « Je ne prends que ce que je comprends,",
    "et je m'en remets à l'auteur pour ce que je ne comprends pas »,",
    "nous lui répondons : « Tu es équitable dans ton propos, mais nous ne craignons pas pour toi",
    "ce que tu ne comprends pas. Car tant que tu ne comprends pas, tu es préservé.",
    # 21 - 25
    "Non, le danger ne vient pas de là ! Notre crainte porte sur ce que tu prétends avoir compris,",
    "alors que tu l'as compris à l'opposé de son véritable sens, et voilà où réside le malheur ! »",
    "Il disait : « C'est de là que vient notre problème. Quant à ce que tu ne comprends pas, louange à Dieu,",
    "il n'y a aucun danger pour toi.",
    "Mais le péril est dans ce que tu crois avoir compris, alors qu'en réalité",
    # 26 - 30
    "tu n'as pas saisi l'intention réelle du Cheikh. Or, ces maîtres,",
    "qu'il s'agisse d'Ibn 'Arabî ou d'autres, ont clairement explicité dans leurs autres ouvrages et poèmes",
    "ce qu'ils entendaient par ces propos gnostiques,",
    "réfutant ainsi tout ce qu'on a pu leur imputer à travers des paroles",
    "dont ils n'ont jamais eu l'intention et qui n'ont jamais traversé leur esprit.",
    # 31 - 35
    "Ils font partie des serviteurs d'élite d'Allah, dotés de la connaissance intime et de l'amour pur.",
    "Cependant, il convient d'être très vigilant face à la lecture de ces livres sans être guidé par un maître authentique.",
    "L'Imam al-Haddad vit un jour son disciple, le Cheikh al-Husayn, lire un livre d'Ibn 'Arabî. Il lui dit :",
    "« Lis-le moi. » Le disciple lui lut l'ouvrage. Dès qu'il l'eut terminé, l'Imam lui dit :",
    "« Écoute bien : à partir d'aujourd'hui, ne rouvre plus ce livre et ne t'y replonge pas. »",
    # 36 - 40
    "Le disciple l'avait lu en présence du maître, et la bénédiction du maître l'avait empêché de mal comprendre.",
    "Mais s'il lisait seul, des doutes et des déviations auraient pu surgir.",
    "Sous la protection du Cheikh, la lecture s'était faite sans aucune ambiguïté.",
    "Mais le Cheikh lui a dit : « Seul, ne t'y replonge plus. »",
    "L'Imam al-Haddad voulait entendre ce que contenait l'ouvrage et en tirer la quintessence.",
    # 41 - 45
    "Quant à celui qui passe son temps à condamner les saints et les élus d'Allah, c'est une calamité majeure",
    "qui frappe certains individus. Même s'il possédait la science des premiers et des derniers, cela ne lui profiterait pas.",
    "Et le plus souvent, ceux qui s'acharnent à attaquer ces maîtres",
    "sont des esprits défaillants en intelligence, en discernement et en science véritable.",
    "Et même si quelqu'un a étudié un peu de logique ou quelques sciences rationnelles,",
    # 46 - 50
    "ne sois pas impressionné par son vernis s'il n'a pas reçu le savoir de maîtres spirituels authentiques.",
    "Ni sa prétendue logique ni sa rationalité ne doivent t'abuser, car la réalité dépasse de loin tout cela !",
    "Ces élus ont plongé au plus profond des sens de la Révélation et du Texte Sacré,",
    "ils ont atteint la moelle par une voie saine, claire, à travers l'effort spirituel et le cheminement.",
    "Comment pourrait-on les comparer à quelqu'un qui a compilé quelques notions disparates,",
    # 51 - 55
    "et qui se permet ensuite de dénigrer les gens parvenus à Allah et les maîtres de la connaissance intime de Dieu ?",
    "Le drame de l'être humain réside dans l'illusion de son intellect et l'orgueil de son savoir.",
    "Allah Tout-Puissant a même repris Son noble serviteur et bien-aimé Moïse — sur lui la paix —",
    "lorsque Moïse pensa qu'il était le plus savant des habitants de la terre.",
    "Allah lui fit alors entreprendre un voyage jusqu'au confluent des deux mers pour y rencontrer al-Khidr.",
    # 56 - 60
    "À la question : « Y a-t-il sur terre quelqu'un de plus savant que toi ? », Moïse répondit : « Non ».",
    "Allah lui reprocha de ne pas Lui avoir rapporté la science et lui dit :",
    "« Notre serviteur al-Khidr en sait plus que toi sur certains aspects. »",
    "C'est-à-dire qu'il y avait des dons divins que Moïse ignorait,",
    "bien que Moïse fût plus savant qu'al-Khidr dans d'autres domaines de la Loi.",
    # 61 - 65
    "Mais Allah aime que Ses serviteurs reconnaissent leur propre mesure et ne rapportent la Science qu'à Lui seul.",
    "C'est ce que les Anges ont exprimé en disant : « Gloire à Toi ! Nous n'avons de savoir que ce que Tu nous as enseigné. »",
    "Si un tel rappel a été adressé au noble Moïse, que dire alors de celui qui s'imagine",
    "plus savant que les plus grands maîtres de son époque, et plus savant que les sommités des siècles passés ?",
    "Celui qui prétend détenir la Vérité absolue tout en étant dans l'égarement le plus total,",
    # 66 - 70
    "sans jamais s'être approché de la Réalité, ni de près ni de loin !",
    "C'est ce qui est arrivé dans l'histoire de la communauté, comme dans le cas d'Ibn Taymiyyah et de ses critiques.",
    "Le savant as-Subkî disait à son sujet : « Il possédait une vive intelligence et une vaste érudition,",
    "mais il n'a pas trouvé de maître spirituel pour le guider. »",
    "Et malgré ses contradictions et ses excès dans la polémique,",
    # 71 - 75
    "Ibn Taymiyyah cherchait parfois à se rattacher à des voies spirituelles :",
    "il vénérait Sayyidunâ 'Abd al-Qâdir al-Jîlânî et glorifiait sa voie soufie.",
    "Cependant, lorsqu'il s'est laissé égarer par l'illusion de sa propre vivacité intellectuelle,",
    "il est tombé dans la contradiction, il s'est mis à critiquer ce qui n'avait pas lieu d'être critiqué,",
    "et il est tombé dans une dureté excessive.",
    # 76 - 82
    "Telle est la mise en garde salutaire : que l'homme ne s'illusionne pas par son propre intellect,",
    "et qu'il respecte le rang sacré des saints et des bien-aimés d'Allah."
]

def main():
    srt_path = "outputs/facebook_1731823678107989.srt"
    entries = parse_srt(srt_path)
    
    # 1. Sauvegarde du texte arabe complet
    ar_text = " ".join([e["text"] for e in entries])
    with open("outputs/facebook_1731823678107989_ar.txt", "w", encoding="utf-8") as f:
        f.write(ar_text)
    
    # 2. Creation du SRT francais
    fr_srt_lines = []
    num_entries = len(entries)
    num_trans = len(translations_fr)
    
    for i, e in enumerate(entries):
        idx = e["index"]
        timing = e["timing"]
        # Map or interpolate translations
        trans_idx = int(i * num_trans / num_entries)
        fr_text = translations_fr[min(trans_idx, num_trans - 1)]
        fr_srt_lines.append(f"{idx}\n{timing}\n{fr_text}\n")
        
    with open("outputs/facebook_1731823678107989_fr.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(fr_srt_lines))
        
    # 3. Sauvegarde de la traduction francaise complete
    full_fr = "\n\n".join([
        "### 1. Le statut des grands maîtres spirituels et la prudence dans la lecture de leurs œuvres\n"
        "En ce qui concerne les paroles des gnostiques et des grands maîtres, certaines passions déforment les propos tenus à leur sujet, comme le rappelle l'Imam al-Haddad dans la Risâlat al-Mu'âwanah. Ces paroles émanent de nos pieux prédécesseurs d'Orient et d'Occident que Dieu a illuminés par la piété, tels que Sayyidunâ Ibn 'Arabî, Sayyidunâ 'Abd al-Karîm al-Jîlî ou 'Abd al-Ghanî an-Nâbulusî. Il est universellement reconnu qu'ils font partie de l'élite de la communauté, des savants vertueux et des connaisseurs de Dieu.\n"
        "Le fait que leurs propos soient ardus pour le commun des mortels et difficiles à appréhender ne diminue en rien leur rang éminent. La mise en garde des maîtres contre la lecture solitaire de ces livres par les débutants est parfaitement légitime : ils craignent que ces textes ne soient interprétés à contre-sens, menant au dévoiement.",

        "### 2. Le dialogue frappant de l'Imam al-Haddad avec le lecteur\n"
        "L'Imam al-Haddad disait à ce propos : Si le lecteur avance : « Je ne prends de ces livres que ce que je comprends, et je m'en remets à l'auteur pour ce que je ne comprends pas », nous lui répondons : « Ton intention paraît juste, mais en réalité, nous ne craignons rien pour toi concernant ce que tu ne comprends pas, car là tu es préservé. Le véritable péril réside dans ce que tu t'imagines avoir compris, alors que tu l'as interprété à l'opposé de l'intention réelle du Cheikh ! »\n"
        "Ces maîtres ont par ailleurs rédigé de nombreux autres traités et poèmes qui explicitent leur doctrine et lavent leur honneur de toutes les dérives qu'on a pu leur attribuer à tort.",

        "### 3. L'anecdote avec le disciple al-Husayn et l'importance du maître\n"
        "L'Imam al-Haddad demanda un jour à son disciple le Cheikh al-Husayn de lui faire la lecture d'un livre d'Ibn 'Arabî. Le disciple lut l'ouvrage en sa présence sans aucune difficulté, car la présence spirituelle du maître le protégeait de toute fausse interprétation. Mais une fois le livre achevé, l'Imam lui ordonna : « À partir d'aujourd'hui, ne rouvre plus jamais cet ouvrage seul. »\n"
        "Car l'homme livré à son seul intellect finit par chuter.",

        "### 4. Le piège de l'orgueil intellectuel et la leçon de Moïse avec al-Khidr\n"
        "S'acharner à critiquer les saints et les élus de Dieu est une tragédie de l'âme. Même si un individu accumulait la science des premiers et des derniers, cela ne lui profiterait guère s'il passe son temps à diffamer les bien-aimés d'Allah. Souvent, ceux qui s'érigent en censeurs ont un entendement limité et manquent d'éducation spirituelle auprès de maîtres vivants. Ne te laisse pas séduire par celui qui manie superficiellement des arguments rationnels ou de logique : la Réalité spirituelle est infiniment plus vaste.\n"
        "Dieu a même repris Son noble Prophète Moïse lorsque celui-ci pensa qu'il était le plus savant de la terre. Dieu lui fit entreprendre un long périple vers le confluent des deux mers pour rencontrer al-Khidr, lui rappelant qu'il existe des sagesses et des dons divins qui échappent même aux plus grands esprits. Si Moïse fut ainsi rappelé à l'humilité, que dire de celui qui, dans son arrogance moderne, s'imagine supérieur aux sommités et aux géants de l'islam ?",

        "### 5. L'exemple d'Ibn Taymiyyah : science sans maître spirituel\n"
        "C'est ce qui est arrivé dans l'histoire, comme chez Ibn Taymiyyah. Le savant as-Subkî disait de lui : « Il avait une immense intelligence et une vaste érudition, mais il n'a pas trouvé de maître spirituel accompli pour le guider. »\n"
        "Bien qu'il vénérait Sayyidunâ 'Abd al-Qâdir al-Jîlânî et respectait sa voie, son excès de confiance en sa propre vivacité intellectuelle l'a parfois conduit à la contradiction, à l'emportement et à la contestation de réalités spirituelles qui le dépassaient."
    ])
    
    with open("outputs/facebook_1731823678107989_fr.txt", "w", encoding="utf-8") as f:
        f.write(full_fr)
        
    print("[SUCCESS] Exported:")
    print(" - outputs/facebook_1731823678107989_ar.txt")
    print(" - outputs/facebook_1731823678107989_fr.srt")
    print(" - outputs/facebook_1731823678107989_fr.txt")

if __name__ == "__main__":
    main()
