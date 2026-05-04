"""Script ponctuel : ajoute les accents français manquants dans README.md.
Les blocs de code (``` ... ```) et le code inline (` ... `) sont préservés tels quels.
Utilise des remplacements par mots entiers (regex \\b) pour éviter les collisions.
"""
import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"

# ---------------------------------------------------------------------------
# Table de substitution : (mot_sans_accent, mot_avec_accent)
# Seul le mot exact (bornes de mots) est remplacé, et seulement s'il ne contient
# pas déjà un accent (on vérifie que le mot en minuscule correspond exactement).
# ---------------------------------------------------------------------------
PAIRS_LOWER = [
    # événement(s)
    ("evenements", "événements"), ("evenement", "événement"),
    # métadonnée(s)
    ("metadonnees", "métadonnées"), ("metadonnee", "métadonnée"),
    # déduplication
    ("deduplication", "déduplication"),
    ("dedupliques", "dédupliqués"), ("deduplique", "dédupliqué"),
    # périmètre
    ("perimetre", "périmètre"),
    # période(s)
    ("periodes", "périodes"), ("periode", "période"),
    # cohérence / cohérent(e)(s)
    ("coherence", "cohérence"),
    ("coherents", "cohérents"), ("coherentes", "cohérentes"),
    ("coherente", "cohérente"), ("coherent", "cohérent"),
    # mécanisme(s)
    ("mecanismes", "mécanismes"), ("mecanisme", "mécanisme"),
    # sémantique(s)
    ("semantiques", "sémantiques"), ("semantique", "sémantique"),
    # métrique(s)
    ("metriques", "métriques"), ("metrique", "métrique"),
    # métier(s)
    ("metiers", "métiers"), ("metier", "métier"),
    # référence(s)
    ("references", "références"), ("reference", "référence"),
    # déictique(s)
    ("deictiques", "déictiques"), ("deictique", "déictique"),
    # requête(s)
    ("requetes", "requêtes"), ("requete", "requête"),
    # réponse(s)
    ("reponses", "réponses"), ("reponse", "réponse"),
    # données / donnée
    ("donnees", "données"), ("donnee", "donnée"),
    # étape(s)
    ("etapes", "étapes"), ("etape", "étape"),
    # détail(s)
    ("details", "détails"), ("detail", "détail"),
    # défaut(s)
    ("defauts", "défauts"), ("defaut", "défaut"),
    # précision(s)
    ("precisions", "précisions"), ("precision", "précision"),
    # régénération / régénéré(e)(s) / régénérer
    ("regeneration", "régénération"),
    ("regenerees", "régénérées"), ("regeneree", "régénérée"),
    ("regeneres", "régénérés"), ("regenerer", "régénérer"),
    ("regenere", "régénéré"),
    # génération / généré(e)(s) / générer / générique(s)
    ("generation", "génération"),
    ("generiques", "génériques"), ("generique", "générique"),
    ("generees", "générées"), ("generee", "générée"),
    ("generes", "générés"), ("generer", "générer"),
    ("genere", "généré"),
    # supérieur(e)(s)
    ("superieures", "supérieures"), ("superieure", "supérieure"),
    ("superieurs", "supérieurs"), ("superieur", "supérieur"),
    # inférieur(e)(s)
    ("inferieures", "inférieures"), ("inferieure", "inférieure"),
    ("inferieurs", "inférieurs"), ("inferieur", "inférieur"),
    # antérieur(e)(s)
    ("anterieures", "antérieures"), ("anterieure", "antérieure"),
    ("anterieurs", "antérieurs"), ("anterieur", "antérieur"),
    # critère(s)
    ("criteres", "critères"), ("critere", "critère"),
    # stratégie(s)
    ("strategies", "stratégies"), ("strategie", "stratégie"),
    # vérification(s) / vérifié(e)(s) / vérifier
    ("verifications", "vérifications"), ("verification", "vérification"),
    ("verifiees", "vérifiées"), ("verifiee", "vérifiée"),
    ("verifies", "vérifiés"), ("verifier", "vérifier"),
    ("verifie", "vérifié"),
    # récupération / récupéré(e)(s) / récupérer
    ("recuperation", "récupération"),
    ("recuperees", "récupérées"), ("recuperee", "récupérée"),
    ("recuperes", "récupérés"), ("recuperer", "récupérer"),
    ("recupere", "récupéré"),
    # intégralité
    ("integralite", "intégralité"),
    # intégration / intégré(e)(s) / intégrer
    ("integration", "intégration"),
    ("integrees", "intégrées"), ("integree", "intégrée"),
    ("integres", "intégrés"), ("integrer", "intégrer"),
    ("integre", "intégré"),
    # réaliser / réalisé(e)(s)
    ("realiser", "réaliser"),
    ("realisees", "réalisées"), ("realisee", "réalisée"),
    ("realises", "réalisés"), ("realise", "réalisé"),
    # dernière(s)
    ("dernieres", "dernières"), ("derniere", "dernière"),
    # première(s)
    ("premieres", "premières"), ("premiere", "première"),
    # déplacement / déplacer / déplacé
    ("deplacement", "déplacement"), ("deplacer", "déplacer"),
    ("deplace", "déplacé"),
    # désactivation / désactiver / désactivé
    ("desactivation", "désactivation"), ("desactiver", "désactiver"),
    ("desactive", "désactivé"),
    # médiane
    ("mediane", "médiane"),
    # précisément / précises / précise
    ("precisement", "précisément"),
    ("precises", "précises"), ("precise", "précise"),
    # éléments / élément
    ("elements", "éléments"), ("element", "élément"),
    # évaluation(s)
    ("evaluations", "évaluations"), ("evaluation", "évaluation"),
    # résultats / résultat
    ("resultats", "résultats"), ("resultat", "résultat"),
    # résumé
    ("resume", "résumé"),
    # démarrage / démarrer
    ("demarrage", "démarrage"), ("demarrer", "démarrer"),
    # détection / détecter / détecté(e)(s)
    ("detection", "détection"), ("detecter", "détecter"),
    ("detectees", "détectées"), ("detectee", "détectée"),
    ("detectes", "détectés"), ("detecte", "détecté"),
    # spécifique(s)
    ("specifiques", "spécifiques"), ("specifique", "spécifique"),
    # nécessaire(s)
    ("necessaires", "nécessaires"), ("necessaire", "nécessaire"),
    # répétition(s)
    ("repetitions", "répétitions"), ("repetition", "répétition"),
    # géographique(s)
    ("geographiques", "géographiques"), ("geographique", "géographique"),
    # présence / présents / présente
    ("presence", "présence"),
    ("presents", "présents"), ("presente", "présente"),
    # dépôt
    ("depot", "dépôt"),
    # système(s)
    ("systemes", "systèmes"), ("systeme", "système"),
    # modèle(s)
    ("modeles", "modèles"), ("modele", "modèle"),
    # catégorie(s)
    ("categories", "catégories"), ("categorie", "catégorie"),
    # réel(le)(s)
    ("reelles", "réelles"), ("reelle", "réelle"),
    ("reels", "réels"), ("reel", "réel"),
    # caractère(s)
    ("caracteres", "caractères"), ("caractere", "caractère"),
    # différent(e)(s)
    ("differents", "différents"), ("differentes", "différentes"),
    ("differente", "différente"), ("different", "différent"),
    # résolution
    ("resolution", "résolution"),
    # présentation
    ("presentation", "présentation"),
    # complexité
    ("complexite", "complexité"),
    # qualité
    ("qualite", "qualité"),
    # capacité
    ("capacite", "capacité"),
    # dépendance(s)
    ("dependances", "dépendances"), ("dependance", "dépendance"),
    # mémoire
    ("memoire", "mémoire"),
    # séquence(s)
    ("sequences", "séquences"), ("sequence", "séquence"),
    # fenêtre(s)
    ("fenetres", "fenêtres"), ("fenetre", "fenêtre"),
    # répartition
    ("repartition", "répartition"),
    # écriture
    ("ecriture", "écriture"),
    # déterministe
    ("deterministe", "déterministe"),
    # représentation
    ("representation", "représentation"),
    # éventuel(le)(s)
    ("eventuelles", "éventuelles"), ("eventuelle", "éventuelle"),
    ("eventuels", "éventuels"), ("eventuel", "éventuel"),
    # créer / créé(e)(s)
    ("creees", "créées"), ("creee", "créée"),
    ("crees", "créés"), ("creer", "créer"),
    ("cree", "créé"),
    # séquentiellement
    ("sequentiellement", "séquentiellement"),
    # sécurité
    ("securite", "sécurité"),
    # associé(e)(s)
    ("associees", "associées"), ("associee", "associée"),
    ("associes", "associés"), ("associe", "associé"),
    # biaisé
    ("biaise", "biaisé"),
    # récent(e)(s)
    ("recents", "récents"), ("recente", "récente"),
    # état(s)
    ("etats", "états"), ("etat", "état"),
    # spécialisé
    ("specialise", "spécialisé"),
    # clé(s)
    ("cles", "clés"), ("cle", "clé"),
    # après
    ("apres", "après"),
    # déjà
    ("deja", "déjà"),
    # paramétrage / paramètre(s)
    ("parametrage", "paramétrage"),
    ("parametres", "paramètres"), ("parametre", "paramètre"),
    # recommandé(e)(s) / recommander
    ("recommandees", "recommandées"), ("recommandee", "recommandée"),
    ("recommandees", "recommandés"), ("recommande", "recommandé"),
    # exécuter / exécution
    ("executer", "exécuter"), ("execution", "exécution"),
    # élargir
    ("elargir", "élargir"),
    # décrire / décrit(e)(s)
    ("decrire", "décrire"), ("decrit", "décrit"),
    # préférence(s) / préféré
    ("preferences", "préférences"), ("preference", "préférence"),
    # informations déjà bien gérées, quelques restes courants
    ("informaticien", "informaticien"),  # pas d'accent, garder tel quel
    # répliquer / réplication
    ("replication", "réplication"),
    # léger(s) / légère(s)
    ("legeres", "légères"), ("legere", "légère"),
    ("legers", "légers"), ("leger", "léger"),
    # déploiement
    ("deploiement", "déploiement"),
    # répertoire(s)
    ("repertoires", "répertoires"), ("repertoire", "répertoire"),
    # données brutes
    ("brutes", "brutes"),  # pas d'accent, idempotent
    # probabilité(s)
    ("probabilites", "probabilités"), ("probabilite", "probabilité"),
    # déjà présent dans le texte via un autre mot → skippé
]

# Génère automatiquement les variantes en Titre (1ère lettre majuscule)
REPLACEMENTS = []
seen = set()
for lower, accented in PAIRS_LOWER:
    for old, new in [(lower, accented),
                     (lower.capitalize(), accented.capitalize())]:
        if old not in seen:
            REPLACEMENTS.append((old, new))
            seen.add(old)


def apply_replacements(text: str) -> str:
    """Remplace les mots sans accent par leur forme accentuée (mots entiers)."""
    for old, new in REPLACEMENTS:
        # Utilise \b (borne de mot) pour ne pas toucher les sous-chaînes
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
    return text


def process(content: str) -> str:
    """Applique les substitutions hors blocs de code et hors code inline."""
    # Découpe : les segments pairs sont du texte libre, les impairs sont du code
    pattern = r'(```[\s\S]*?```|`[^`\n]+`)'
    parts = re.split(pattern, content)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(apply_replacements(part))
        else:
            out.append(part)  # code block inchangé
    return "".join(out)


def main() -> None:
    original = README.read_text(encoding="utf-8")
    updated = process(original)
    if updated == original:
        print("Aucune modification nécessaire.")
        return
    README.write_text(updated, encoding="utf-8")
    # Résumé des changements
    orig_lines = original.splitlines()
    new_lines = updated.splitlines()
    changed = sum(1 for a, b in zip(orig_lines, new_lines) if a != b)
    print(f"✓ {changed} ligne(s) modifiée(s) dans {README.name}")


if __name__ == "__main__":
    main()
