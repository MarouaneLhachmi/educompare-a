/* =========================================================================
   EduCompare AI — Interactions de la couche de présentation
   -------------------------------------------------------------------------
   Tout est piloté par des attributs `data-*` posés dans les templates Jinja :
   aucun identifiant en dur, aucun code métier ici.
   ========================================================================= */
(function () {
    "use strict";

    const $ = (sel, racine) => (racine || document).querySelector(sel);
    const $$ = (sel, racine) => Array.from((racine || document).querySelectorAll(sel));
    const mouvementReduit = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /* ------------------------------------------------------------------ */
    /* Thème clair / sombre (persisté dans le navigateur)                  */
    /* ------------------------------------------------------------------ */
    const THEME_CLE = "educompare-theme";

    function appliquerTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        $$("[data-bascule-theme]").forEach((b) => {
            b.textContent = theme === "dark" ? "☀️" : "🌙";
            b.setAttribute("aria-label", theme === "dark" ? "Passer en thème clair" : "Passer en thème sombre");
        });
    }

    function initTheme() {
        const enregistre = localStorage.getItem(THEME_CLE);
        const systeme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        appliquerTheme(enregistre || systeme);

        $$("[data-bascule-theme]").forEach((bouton) => {
            bouton.addEventListener("click", () => {
                const actuel = document.documentElement.getAttribute("data-theme");
                const suivant = actuel === "dark" ? "light" : "dark";
                localStorage.setItem(THEME_CLE, suivant);
                appliquerTheme(suivant);
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* En-tête : ombre au défilement                                       */
    /* ------------------------------------------------------------------ */
    function initEntete() {
        const entete = $(".entete");
        if (!entete) return;
        const majEntete = () => entete.classList.toggle("defile", window.scrollY > 8);
        majEntete();
        window.addEventListener("scroll", majEntete, { passive: true });
    }

    /* ------------------------------------------------------------------ */
    /* Menus déroulants                                                    */
    /* ------------------------------------------------------------------ */
    function initMenus() {
        $$("[data-menu-declencheur]").forEach((declencheur) => {
            const menu = $("#" + declencheur.getAttribute("data-menu-declencheur"));
            if (!menu) return;
            declencheur.addEventListener("click", (evt) => {
                evt.stopPropagation();
                const ouvert = menu.classList.contains("ouvert");
                $$(".menu-deroulant.ouvert").forEach((m) => m.classList.remove("ouvert"));
                menu.classList.toggle("ouvert", !ouvert);
                declencheur.setAttribute("aria-expanded", String(!ouvert));
            });
        });
        document.addEventListener("click", () => {
            $$(".menu-deroulant.ouvert").forEach((m) => m.classList.remove("ouvert"));
            $$("[data-menu-declencheur]").forEach((d) => d.setAttribute("aria-expanded", "false"));
        });
        document.addEventListener("keydown", (evt) => {
            if (evt.key === "Escape") $$(".menu-deroulant.ouvert").forEach((m) => m.classList.remove("ouvert"));
        });
    }

    /* ------------------------------------------------------------------ */
    /* Messages flash : fermeture manuelle + disparition automatique       */
    /* ------------------------------------------------------------------ */
    function fermerFlash(flash) {
        flash.classList.add("sortie");
        setTimeout(() => flash.remove(), 350);
    }

    function initFlash() {
        $$(".flash").forEach((flash, index) => {
            const bouton = $(".flash-fermer", flash);
            if (bouton) bouton.addEventListener("click", () => fermerFlash(flash));
            setTimeout(() => flash.parentNode && fermerFlash(flash), 7000 + index * 600);
        });
    }

    /* ------------------------------------------------------------------ */
    /* Apparition progressive au défilement                                */
    /* ------------------------------------------------------------------ */
    function initRevelation() {
        const cibles = $$(".reveler");
        if (!cibles.length) return;
        if (mouvementReduit || !("IntersectionObserver" in window)) {
            cibles.forEach((c) => c.classList.add("visible"));
            return;
        }
        const observateur = new IntersectionObserver(
            (entrees) => {
                entrees.forEach((entree, i) => {
                    if (!entree.isIntersecting) return;
                    setTimeout(() => entree.target.classList.add("visible"), i * 70);
                    observateur.unobserve(entree.target);
                });
            },
            { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
        );
        cibles.forEach((c) => observateur.observe(c));
    }

    /* ------------------------------------------------------------------ */
    /* Compteurs animés  —  <span data-compteur="87.5" data-decimales="1"> */
    /* ------------------------------------------------------------------ */
    function animerCompteur(element) {
        const cible = parseFloat(element.getAttribute("data-compteur")) || 0;
        const decimales = parseInt(element.getAttribute("data-decimales") || "0", 10);
        const suffixe = element.getAttribute("data-suffixe") || "";
        // Onglet masqué ou animations désactivées : on affiche la valeur finale
        // sans animer (requestAnimationFrame est suspendu sur un onglet caché).
        if (mouvementReduit || document.hidden) {
            element.textContent = cible.toFixed(decimales) + suffixe;
            return;
        }
        const duree = 1400;
        const debut = performance.now();
        function etape(maintenant) {
            const t = Math.min((maintenant - debut) / duree, 1);
            const adouci = 1 - Math.pow(1 - t, 3);
            element.textContent = (cible * adouci).toFixed(decimales) + suffixe;
            if (t < 1) requestAnimationFrame(etape);
        }
        requestAnimationFrame(etape);
    }

    function initCompteurs() {
        const compteurs = $$("[data-compteur]");
        if (!compteurs.length) return;
        if (!("IntersectionObserver" in window)) {
            compteurs.forEach(animerCompteur);
            return;
        }
        const observateur = new IntersectionObserver(
            (entrees) => {
                entrees.forEach((entree) => {
                    if (!entree.isIntersecting) return;
                    animerCompteur(entree.target);
                    observateur.unobserve(entree.target);
                });
            },
            { threshold: 0.4 }
        );
        compteurs.forEach((c) => observateur.observe(c));
    }

    /* ------------------------------------------------------------------ */
    /* Barres de progression  —  <div class="barre-remplissage" data-pct>  */
    /* ------------------------------------------------------------------ */
    function initBarres() {
        const barres = $$("[data-pct]");
        if (!barres.length) return;
        const remplir = (barre) => {
            const pct = Math.max(0, Math.min(100, parseFloat(barre.getAttribute("data-pct")) || 0));
            requestAnimationFrame(() => { barre.style.width = pct + "%"; });
        };
        if (!("IntersectionObserver" in window)) { barres.forEach(remplir); return; }
        const observateur = new IntersectionObserver(
            (entrees) => entrees.forEach((e) => {
                if (!e.isIntersecting) return;
                remplir(e.target);
                observateur.unobserve(e.target);
            }),
            { threshold: 0.25 }
        );
        barres.forEach((b) => observateur.observe(b));
    }

    /* ------------------------------------------------------------------ */
    /* Jauges circulaires  —  <svg><circle class="jauge-valeur" data-jauge> */
    /* ------------------------------------------------------------------ */
    function couleurScore(valeur) {
        if (valeur >= 80) return "#16a34a";
        if (valeur >= 65) return "#4f6df5";
        if (valeur >= 50) return "#f59e0b";
        if (valeur >= 35) return "#f97316";
        return "#ef4444";
    }

    function initJauges() {
        $$("[data-jauge]").forEach((cercle) => {
            const valeur = Math.max(0, Math.min(100, parseFloat(cercle.getAttribute("data-jauge")) || 0));
            const rayon = cercle.r.baseVal.value;
            const circonference = 2 * Math.PI * rayon;
            cercle.style.setProperty("--circ", circonference);
            cercle.style.strokeDasharray = circonference;
            cercle.style.strokeDashoffset = circonference;
            cercle.style.stroke = cercle.getAttribute("data-couleur") || couleurScore(valeur);

            const remplir = () => {
                requestAnimationFrame(() => {
                    cercle.style.strokeDashoffset = circonference * (1 - valeur / 100);
                });
            };
            if ("IntersectionObserver" in window) {
                const obs = new IntersectionObserver((entrees) => {
                    entrees.forEach((e) => { if (e.isIntersecting) { remplir(); obs.disconnect(); } });
                }, { threshold: 0.3 });
                obs.observe(cercle);
            } else {
                remplir();
            }
        });
    }

    /* ------------------------------------------------------------------ */
    /* Radar des indicateurs — <svg data-radar='[{"libelle","valeur"}]'>   */
    /* ------------------------------------------------------------------ */
    function initRadars() {
        $$("[data-radar]").forEach((svg) => {
            let donnees;
            try { donnees = JSON.parse(svg.getAttribute("data-radar")); } catch (e) { return; }
            if (!Array.isArray(donnees) || donnees.length < 3) return;

            const taille = 300, centre = taille / 2, rayonMax = 96;
            const n = donnees.length;
            const NS = "http://www.w3.org/2000/svg";
            svg.setAttribute("viewBox", `0 0 ${taille} ${taille}`);
            svg.innerHTML = "";

            const point = (i, ratio) => {
                const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
                return [centre + Math.cos(angle) * rayonMax * ratio, centre + Math.sin(angle) * rayonMax * ratio];
            };

            // Toile de fond : anneaux concentriques
            [0.25, 0.5, 0.75, 1].forEach((ratio) => {
                const pts = donnees.map((_, i) => point(i, ratio).join(",")).join(" ");
                const poly = document.createElementNS(NS, "polygon");
                poly.setAttribute("points", pts);
                poly.setAttribute("class", "grille-radar");
                svg.appendChild(poly);
            });

            // Axes et libellés
            donnees.forEach((d, i) => {
                const [x, y] = point(i, 1);
                const axe = document.createElementNS(NS, "line");
                axe.setAttribute("x1", centre); axe.setAttribute("y1", centre);
                axe.setAttribute("x2", x); axe.setAttribute("y2", y);
                axe.setAttribute("class", "axe");
                svg.appendChild(axe);

                const [lx, ly] = point(i, 1.25);
                const texte = document.createElementNS(NS, "text");
                texte.setAttribute("x", lx);
                texte.setAttribute("y", ly);
                texte.setAttribute("text-anchor", lx > centre + 6 ? "start" : lx < centre - 6 ? "end" : "middle");
                texte.setAttribute("dominant-baseline", "middle");
                (d.libelle || "").split(" ").slice(0, 3).forEach((mot, ligne) => {
                    const tspan = document.createElementNS(NS, "tspan");
                    tspan.setAttribute("x", lx);
                    tspan.setAttribute("dy", ligne === 0 ? 0 : 10);
                    tspan.textContent = mot;
                    texte.appendChild(tspan);
                });
                svg.appendChild(texte);
            });

            // Zone des valeurs mesurées
            const zone = document.createElementNS(NS, "polygon");
            zone.setAttribute("points", donnees.map((d, i) => point(i, Math.max(0.02, Math.min(1, d.valeur))).join(",")).join(" "));
            zone.setAttribute("class", "zone");
            svg.appendChild(zone);

            donnees.forEach((d, i) => {
                const [x, y] = point(i, Math.max(0.02, Math.min(1, d.valeur)));
                const cercle = document.createElementNS(NS, "circle");
                cercle.setAttribute("cx", x); cercle.setAttribute("cy", y); cercle.setAttribute("r", 3.2);
                cercle.setAttribute("class", "point");
                const titre = document.createElementNS(NS, "title");
                titre.textContent = `${d.libelle} : ${Number(d.valeur).toFixed(2)}`;
                cercle.appendChild(titre);
                svg.appendChild(cercle);
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Anneau de répartition — <svg data-anneau='[{valeur,couleur,libelle}]'> */
    /* ------------------------------------------------------------------ */
    function initAnneaux() {
        $$("[data-anneau]").forEach((svg) => {
            let segments;
            try { segments = JSON.parse(svg.getAttribute("data-anneau")); } catch (e) { return; }
            const total = segments.reduce((s, x) => s + (x.valeur || 0), 0) || 1;
            const NS = "http://www.w3.org/2000/svg";
            const rayon = 54, circonference = 2 * Math.PI * rayon;
            svg.setAttribute("viewBox", "0 0 140 140");
            svg.innerHTML = "";

            const fond = document.createElementNS(NS, "circle");
            fond.setAttribute("cx", 70); fond.setAttribute("cy", 70); fond.setAttribute("r", rayon);
            fond.setAttribute("stroke", "var(--surface-3)");
            svg.appendChild(fond);

            let decalage = 0;
            segments.forEach((segment, i) => {
                const portion = (segment.valeur || 0) / total;
                const arc = document.createElementNS(NS, "circle");
                arc.setAttribute("cx", 70); arc.setAttribute("cy", 70); arc.setAttribute("r", rayon);
                arc.setAttribute("stroke", segment.couleur);
                arc.setAttribute("stroke-dasharray", `0 ${circonference}`);
                arc.setAttribute("transform", `rotate(${-90 + decalage * 360} 70 70)`);
                arc.setAttribute("stroke-linecap", "butt");
                const titre = document.createElementNS(NS, "title");
                titre.textContent = `${segment.libelle} : ${segment.valeur}`;
                arc.appendChild(titre);
                svg.appendChild(arc);
                setTimeout(() => {
                    arc.setAttribute("stroke-dasharray", `${portion * circonference} ${circonference}`);
                }, 120 + i * 180);
                decalage += portion;
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Courbe de progression                                               */
    /* <svg data-courbe='[{"x":0,"y":0.31,"libelle":"Séance 0"}, ...]'>     */
    /* ------------------------------------------------------------------ */
    function initCourbes() {
        $$("[data-courbe]").forEach((svg) => {
            let points;
            try { points = JSON.parse(svg.getAttribute("data-courbe")); } catch (e) { return; }
            if (!Array.isArray(points) || points.length < 2) return;

            const NS = "http://www.w3.org/2000/svg";
            const L = 640, H = 220;
            const marge = { haut: 18, bas: 30, gauche: 42, droite: 14 };
            const largeur = L - marge.gauche - marge.droite;
            const hauteur = H - marge.haut - marge.bas;

            const xs = points.map((p) => p.x);
            const ys = points.map((p) => p.y);
            const xMin = Math.min(...xs), xMax = Math.max(...xs);
            // L'échelle verticale est élargie de 15 % autour des valeurs
            // observées : sur un gain de quelques centièmes, un axe 0–1
            // écraserait complètement la courbe.
            const marge_y = Math.max(0.04, (Math.max(...ys) - Math.min(...ys)) * 0.35);
            const yMin = Math.max(0, Math.min(...ys) - marge_y);
            const yMax = Math.min(1, Math.max(...ys) + marge_y);

            const px = (x) => marge.gauche + ((x - xMin) / (xMax - xMin || 1)) * largeur;
            const py = (y) => marge.haut + (1 - (y - yMin) / (yMax - yMin || 1)) * hauteur;

            svg.setAttribute("viewBox", `0 0 ${L} ${H}`);
            svg.setAttribute("preserveAspectRatio", "none");
            svg.innerHTML = "";

            const defs = document.createElementNS(NS, "defs");
            defs.innerHTML =
                '<linearGradient id="degradeCourbe" x1="0" y1="0" x2="0" y2="1">' +
                '<stop offset="0%" stop-color="#4f6df5"/>' +
                '<stop offset="100%" stop-color="#4f6df5" stop-opacity="0"/></linearGradient>';
            svg.appendChild(defs);

            // Grille horizontale + graduations
            for (let i = 0; i <= 4; i++) {
                const valeur = yMin + ((yMax - yMin) * i) / 4;
                const y = py(valeur);
                const ligne = document.createElementNS(NS, "line");
                ligne.setAttribute("x1", marge.gauche); ligne.setAttribute("x2", L - marge.droite);
                ligne.setAttribute("y1", y); ligne.setAttribute("y2", y);
                ligne.setAttribute("class", "grille-h");
                svg.appendChild(ligne);

                const texte = document.createElementNS(NS, "text");
                texte.setAttribute("x", marge.gauche - 8);
                texte.setAttribute("y", y + 3);
                texte.setAttribute("text-anchor", "end");
                texte.textContent = (valeur * 100).toFixed(0) + "%";
                svg.appendChild(texte);
            }

            const chemin = points.map((p, i) => `${i ? "L" : "M"}${px(p.x)},${py(p.y)}`).join(" ");

            const aire = document.createElementNS(NS, "path");
            aire.setAttribute("d", `${chemin} L${px(xMax)},${py(yMin)} L${px(xMin)},${py(yMin)} Z`);
            aire.setAttribute("class", "aire");
            svg.appendChild(aire);

            const trace = document.createElementNS(NS, "path");
            trace.setAttribute("d", chemin);
            trace.setAttribute("class", "trace");
            svg.appendChild(trace);
            const longueur = trace.getTotalLength ? trace.getTotalLength() : 1000;
            trace.style.setProperty("--long", longueur);

            points.forEach((p, i) => {
                const jalon = document.createElementNS(NS, "circle");
                jalon.setAttribute("cx", px(p.x));
                jalon.setAttribute("cy", py(p.y));
                jalon.setAttribute("r", 3.5);
                jalon.setAttribute("class", "jalon");
                const titre = document.createElementNS(NS, "title");
                titre.textContent = `${p.libelle || "Séance " + p.x} : ${(p.y * 100).toFixed(1)} %`;
                jalon.appendChild(titre);
                svg.appendChild(jalon);

                if (i === 0 || i === points.length - 1) {
                    const legende = document.createElementNS(NS, "text");
                    legende.setAttribute("x", px(p.x));
                    legende.setAttribute("y", H - 10);
                    legende.setAttribute("text-anchor", i === 0 ? "start" : "end");
                    legende.textContent = i === 0 ? "Départ" : `Séance ${p.x}`;
                    svg.appendChild(legende);
                }
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Barres du profil de Bloom                                           */
    /* ------------------------------------------------------------------ */
    function initBloom() {
        $$("[data-part]").forEach((element) => {
            const pct = Math.max(0, Math.min(100, parseFloat(element.getAttribute("data-part")) || 0));
            requestAnimationFrame(() => { element.style.width = pct + "%"; });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Zone de dépôt de fichier (glisser-déposer)                          */
    /* ------------------------------------------------------------------ */
    function initZoneDepot() {
        const zone = $("[data-zone-depot]");
        if (!zone) return;
        const entree = $("input[type=file]", zone) || $("#" + zone.getAttribute("data-zone-depot"));
        if (!entree) return;
        const nomAffiche = $("[data-nom-fichier]", zone);
        const tailleAffichee = $("[data-taille-fichier]", zone);

        const formaterTaille = (octets) => {
            const unites = ["o", "Ko", "Mo", "Go"];
            let i = 0, valeur = octets;
            while (valeur >= 1024 && i < unites.length - 1) { valeur /= 1024; i++; }
            return `${valeur.toFixed(i === 0 ? 0 : 1)} ${unites[i]}`;
        };

        const afficher = () => {
            const fichier = entree.files && entree.files[0];
            if (!fichier) { zone.classList.remove("rempli"); return; }
            zone.classList.add("rempli");
            if (nomAffiche) nomAffiche.textContent = fichier.name;
            if (tailleAffichee) tailleAffichee.textContent = formaterTaille(fichier.size);
        };

        zone.addEventListener("click", (evt) => {
            if (evt.target !== entree) { evt.preventDefault(); entree.click(); }
        });
        entree.addEventListener("change", afficher);

        ["dragenter", "dragover"].forEach((evt) =>
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("survol"); })
        );
        ["dragleave", "drop"].forEach((evt) =>
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("survol"); })
        );
        zone.addEventListener("drop", (e) => {
            if (e.dataTransfer && e.dataTransfer.files.length) {
                entree.files = e.dataTransfer.files;
                afficher();
            }
        });
        afficher();
    }

    /* ------------------------------------------------------------------ */
    /* Formulaire de dépôt : état « envoi en cours »                       */
    /* ------------------------------------------------------------------ */
    function initSoumission() {
        $$("[data-formulaire-analyse]").forEach((formulaire) => {
            formulaire.addEventListener("submit", () => {
                const bouton = $("[data-bouton-soumettre]", formulaire);
                if (!bouton) return;
                bouton.disabled = true;
                bouton.dataset.libelleInitial = bouton.innerHTML;
                bouton.innerHTML = '<span class="ic">⏳</span> Envoi du document…';
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Onglets                                                             */
    /* ------------------------------------------------------------------ */
    function initOnglets() {
        $$("[data-groupe-onglets]").forEach((groupe) => {
            const onglets = $$("[data-onglet]", groupe);
            const panneaux = $$("[data-panneau]", groupe.parentElement || document);
            onglets.forEach((onglet) => {
                onglet.addEventListener("click", () => {
                    const cible = onglet.getAttribute("data-onglet");
                    onglets.forEach((o) => o.classList.toggle("actif", o === onglet));
                    panneaux.forEach((p) => p.classList.toggle("actif", p.getAttribute("data-panneau") === cible));
                });
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Accordéons                                                          */
    /* ------------------------------------------------------------------ */
    function initAccordeons() {
        $$(".accordeon-tete").forEach((tete) => {
            tete.addEventListener("click", () => {
                const accordeon = tete.closest(".accordeon");
                if (accordeon) accordeon.classList.toggle("ouvert");
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Filtrage instantané des tableaux                                    */
    /* ------------------------------------------------------------------ */
    function initFiltreTableau() {
        $$("[data-filtre-tableau]").forEach((champ) => {
            const table = $("#" + champ.getAttribute("data-filtre-tableau"));
            if (!table) return;
            const compteur = $("[data-compteur-lignes]");
            champ.addEventListener("input", () => {
                const terme = champ.value.trim().toLowerCase();
                let visibles = 0;
                $$("tbody tr", table).forEach((ligne) => {
                    const correspond = !terme || ligne.textContent.toLowerCase().includes(terme);
                    ligne.style.display = correspond ? "" : "none";
                    if (correspond) visibles++;
                });
                if (compteur) compteur.textContent = visibles;
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Sélecteur dynamique des référentiels (matière / niveau -> pays)      */
    /* ------------------------------------------------------------------ */
    function initSelecteurPays() {
        const conteneur = $("[data-pays-conteneur]");
        const selMatiere = $("[data-select-matiere]");
        const selNiveau = $("[data-select-niveau]");
        if (!conteneur || !selMatiere || !selNiveau) return;

        async function recharger() {
            const url = `${conteneur.getAttribute("data-pays-conteneur")}?matiere=${encodeURIComponent(selMatiere.value)}&niveau=${encodeURIComponent(selNiveau.value)}`;
            conteneur.style.opacity = ".45";
            try {
                const reponse = await fetch(url);
                const donnees = await reponse.json();
                conteneur.innerHTML = donnees.pays
                    .map(
                        (p) => `
                    <label class="carte-pays">
                        <input type="checkbox" name="pays" value="${p.code}" checked>
                        <span class="drapeau">${p.drapeau || "🏳️"}</span>
                        <span>
                            <span class="nom">${p.pays}</span><br>
                            <span class="meta">${p.nb_notions} notions · ${p.referentiel}</span>
                        </span>
                        <span class="coche">✓</span>
                    </label>`
                    )
                    .join("");
            } catch (e) {
                /* En cas d'échec réseau, la sélection précédente reste utilisable. */
            } finally {
                conteneur.style.opacity = "1";
            }
        }

        selMatiere.addEventListener("change", recharger);
        selNiveau.addEventListener("change", recharger);
    }

    /* ------------------------------------------------------------------ */
    /* Confirmation avant action destructrice                              */
    /* ------------------------------------------------------------------ */
    /* ------------------------------------------------------------------ */
    /* Confirmation des actions destructives                               */
    /* ------------------------------------------------------------------ */
    /* Remplace window.confirm : non stylable, brutal, et surtout incapable de
       rappeler ce qui va être détruit ET ce qui sera conservé. La distinction
       est ce qui manque le plus au moment de décider — supprimer un programme
       n'efface pas les analyses, et l'utilisateur doit le savoir avant de
       cliquer, pas après.

       Le formulaire reste un POST classique : la modale ne fait qu'intercepter
       la soumission. Sans JavaScript, l'action fonctionne toujours. */
    function initConfirmations() {
        const formulaires = $$("[data-modale-titre], [data-confirmer]");
        if (!formulaires.length) return;

        let superposition = null;
        let formulaireEnAttente = null;
        let elementDeclencheur = null;

        function construire() {
            superposition = document.createElement("div");
            superposition.className = "superposition";
            superposition.setAttribute("role", "dialog");
            superposition.setAttribute("aria-modal", "true");
            superposition.innerHTML = `
                <div class="modale">
                    <div class="modale-icone" data-modale-icone>⚠</div>
                    <h2 class="modale-titre" data-modale-titre></h2>
                    <p class="modale-message" data-modale-message></p>
                    <div class="modale-conserve" data-modale-conserve hidden></div>
                    <div class="modale-actions">
                        <button type="button" class="btn btn-secondaire btn-sm" data-modale-annuler>Annuler</button>
                        <button type="button" class="btn btn-danger btn-sm" data-modale-valider>Confirmer</button>
                    </div>
                </div>`;
            document.body.appendChild(superposition);

            superposition.addEventListener("click", (evt) => {
                if (evt.target === superposition) fermer();
            });
            $("[data-modale-annuler]", superposition).addEventListener("click", fermer);
            $("[data-modale-valider]", superposition).addEventListener("click", () => {
                const formulaire = formulaireEnAttente;
                fermer();
                // `requestSubmit` déclencherait de nouveau notre écouteur :
                // on marque le formulaire comme déjà confirmé.
                if (formulaire) {
                    formulaire.dataset.confirme = "1";
                    formulaire.submit();
                }
            });
            document.addEventListener("keydown", (evt) => {
                if (evt.key === "Escape" && superposition.classList.contains("ouverte")) fermer();
            });
        }

        function fermer() {
            superposition.classList.remove("ouverte");
            formulaireEnAttente = null;
            if (elementDeclencheur) elementDeclencheur.focus();
        }

        function ouvrir(formulaire) {
            if (!superposition) construire();
            formulaireEnAttente = formulaire;

            const jeu = formulaire.dataset;
            const simple = !jeu.modaleTitre;
            $("[data-modale-icone]", superposition).textContent = jeu.modaleIcone || "⚠";
            $("[data-modale-titre]", superposition).textContent =
                jeu.modaleTitre || "Confirmer cette action";
            $("[data-modale-message]", superposition).textContent =
                jeu.modaleMessage || jeu.confirmer || "";

            const conserve = $("[data-modale-conserve]", superposition);
            if (jeu.modaleConserve) {
                conserve.textContent = jeu.modaleConserve;
                conserve.hidden = false;
            } else {
                conserve.hidden = true;
            }

            const valider = $("[data-modale-valider]", superposition);
            valider.textContent = jeu.modaleConfirmation || (simple ? "Confirmer" : "Confirmer");

            superposition.classList.add("ouverte");
            valider.focus();
        }

        formulaires.forEach((formulaire) => {
            formulaire.addEventListener("submit", (evt) => {
                if (formulaire.dataset.confirme === "1") return;
                evt.preventDefault();
                elementDeclencheur = document.activeElement;
                ouvrir(formulaire);
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Retour enseignant (mode ombre) : dépose une étiquette sans quitter  */
    /* la page ni changer quoi que ce soit au rapport affiché.             */
    /* ------------------------------------------------------------------ */
    function initRetours() {
        const widgets = $$("[data-retour]");
        if (!widgets.length) return;

        widgets.forEach((widget) => {
            const url = widget.getAttribute("data-retour");
            const type = widget.getAttribute("data-retour-type");
            const cleNotion = widget.getAttribute("data-retour-cle");
            const confirmation = $(".retour-confirmation", widget);
            let enCours = false;

            $$(".retour-bouton", widget).forEach((bouton) => {
                bouton.addEventListener("click", async () => {
                    if (enCours) return;
                    enCours = true;
                    $$(".retour-bouton", widget).forEach((b) => (b.disabled = true));

                    try {
                        const reponse = await fetch(url, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                type,
                                cle_notion: cleNotion,
                                valeur: bouton.getAttribute("data-valeur"),
                            }),
                        });
                        if (!reponse.ok) throw new Error("statut " + reponse.status);

                        $$(".retour-bouton", widget).forEach((b) => b.classList.remove("actif"));
                        bouton.classList.add("actif");
                        if (confirmation) {
                            confirmation.textContent = "Merci, c'est noté.";
                            confirmation.classList.add("visible");
                        }
                    } catch (erreur) {
                        if (confirmation) {
                            confirmation.textContent = "Échec de l'envoi, réessayez.";
                            confirmation.style.color = "var(--danger)";
                            confirmation.classList.add("visible");
                        }
                    } finally {
                        $$(".retour-bouton", widget).forEach((b) => (b.disabled = false));
                        enCours = false;
                    }
                });
            });
        });
    }

    /* ------------------------------------------------------------------ */
    /* Écran de suivi : interrogation périodique de l'avancement           */
    /* ------------------------------------------------------------------ */
    function initSuivi() {
        const racine = $("[data-suivi]");
        if (!racine) return;

        const url = racine.getAttribute("data-suivi");
        const barre = $("[data-suivi-barre]", racine);
        const pourcentage = $("[data-suivi-pourcentage]", racine);
        const message = $("[data-suivi-message]", racine);
        const restant = $("[data-suivi-restant]", racine);
        const journal = $("[data-suivi-journal]", racine);
        const etapes = $("[data-suivi-etapes]", racine);

        const LIBELLES = {
            en_attente: "En attente", en_cours: "En cours…",
            termine: "Terminé", repli: "Repli activé", echec: "Échec",
        };

        let dernierJournal = 0;
        let intervalle = 1200;
        let arrete = false;

        async function interroger() {
            if (arrete) return;
            try {
                const reponse = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
                if (!reponse.ok) throw new Error("statut " + reponse.status);
                const etat = await reponse.json();

                const pct = Math.round(etat.pourcentage || 0);
                if (barre) barre.style.width = pct + "%";
                if (pourcentage) pourcentage.textContent = pct;
                if (message) message.textContent = etat.message || "";

                // Temps restant estimé par le modèle de prédiction de durée,
                // corrigé du rythme réellement observé depuis le début.
                if (restant) {
                    const secondes = etat.temps_restant_s;
                    if (etat.statut === "EN_COURS" && secondes > 0) {
                        const minutes = Math.floor(secondes / 60);
                        const reste = Math.round(secondes % 60);
                        restant.textContent =
                            "⏱ environ " +
                            (minutes ? `${minutes} min ${reste} s` : `${reste} s`) +
                            " restantes";
                        restant.style.display = "";
                    } else {
                        restant.style.display = "none";
                    }
                }

                if (etapes && Array.isArray(etat.agents)) {
                    etat.agents.forEach((agent) => {
                        const ligne = $(`[data-agent="${agent.cle}"]`, etapes);
                        if (!ligne) return;
                        ligne.className = "etape " + agent.statut;
                        const statut = $("[data-agent-statut]", ligne);
                        if (statut) {
                            statut.textContent =
                                LIBELLES[agent.statut] + (agent.duree_s ? ` · ${agent.duree_s}s` : "");
                        }
                        const detail = $("[data-agent-detail]", ligne);
                        if (detail && agent.detail) detail.textContent = agent.detail;
                    });
                }

                if (journal && Array.isArray(etat.journal)) {
                    etat.journal.slice(dernierJournal).forEach((entree) => {
                        const ligne = document.createElement("div");
                        ligne.className = "journal-ligne " + entree.niveau;
                        ligne.innerHTML = `<span class="h">${entree.horodatage}</span><span class="m"></span>`;
                        $(".m", ligne).textContent = entree.message;
                        journal.appendChild(ligne);
                    });
                    if (etat.journal.length > dernierJournal) {
                        dernierJournal = etat.journal.length;
                        journal.scrollTop = journal.scrollHeight;
                    }
                }

                if (etat.statut === "TERMINEE") {
                    arrete = true;
                    if (message) message.textContent = "Analyse terminée — ouverture du rapport…";
                    setTimeout(() => { window.location.href = etat.url_rapport; }, 900);
                    return;
                }
                if (etat.statut === "ECHEC") {
                    arrete = true;
                    if (message) message.textContent = etat.erreur || "L'analyse a échoué.";
                    racine.classList.add("en-echec");
                    setTimeout(() => window.location.reload(), 1500);
                    return;
                }
                intervalle = 1200;
            } catch (e) {
                // Ralentissement progressif en cas d'erreur réseau, sans dépasser 8 s.
                intervalle = Math.min(intervalle * 1.6, 8000);
            }
            setTimeout(interroger, intervalle);
        }

        interroger();
    }

    /* ------------------------------------------------------------------ */
    /* Filet de sécurité                                                   */
    /* Sur certains environnements (onglet jamais peint, navigateur        */
    /* embarqué, impression), IntersectionObserver ne se déclenche pas.    */
    /* Après un court délai, on force l'affichage de tout ce qui se trouve */
    /* déjà dans la fenêtre : la page ne doit jamais rester vide.          */
    /* ------------------------------------------------------------------ */
    function dansLaFenetre(element) {
        const r = element.getBoundingClientRect();
        return r.top < window.innerHeight + 200 && r.bottom > -200;
    }

    function balayerFenetre() {
        $$(".reveler:not(.visible)").forEach((e) => { if (dansLaFenetre(e)) e.classList.add("visible"); });
        $$("[data-compteur]").forEach((e) => {
            if (e.textContent.trim() === "0" && dansLaFenetre(e)) animerCompteur(e);
        });
        $$("[data-pct]").forEach((e) => {
            if (!e.style.width && dansLaFenetre(e)) {
                e.style.width = Math.max(0, Math.min(100, parseFloat(e.getAttribute("data-pct")) || 0)) + "%";
            }
        });
    }

    function filetDeSecurite() {
        setTimeout(balayerFenetre, 1600);
        // Une page ouverte dans un onglet d'arrière-plan ne déclenche ni
        // IntersectionObserver ni requestAnimationFrame : on rebalaie au retour.
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) setTimeout(balayerFenetre, 200);
        });
    }

    /* ------------------------------------------------------------------ */
    /* Initialisation                                                      */
    /* ------------------------------------------------------------------ */
    function init() {
        initTheme();
        initEntete();
        initMenus();
        initFlash();
        initRevelation();
        initCompteurs();
        initBarres();
        initJauges();
        initRadars();
        initAnneaux();
        initCourbes();
        initBloom();
        initZoneDepot();
        initSoumission();
        initOnglets();
        initAccordeons();
        initFiltreTableau();
        initSelecteurPays();
        initConfirmations();
        initRetours();
        initSuivi();
        filetDeSecurite();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
