# Cierre coordinado de la entrega

El docente aprobó usar varios repositorios, según la confirmación del equipo. Esta entrega utiliza tres: [NLP](https://github.com/topicos-proyecto/clasificador-tickets-nlp), [backend](https://github.com/topicos-proyecto/clasificador-backend) y [frontend](https://github.com/topicos-proyecto/clasificador-frontend). La etiqueta `entrega-final` debe identificar el commit evaluado de `main` en cada uno.

## Documentos y presentación

- [Informe técnico](INFORME_TECNICO.md): problema, dataset, NLP, comparación, arquitectura, límites y declaración de asistencia de IA.
- [Contribuciones](../CONTRIBUCIONES.md): nombres registrados en Git, aportes y enlaces a commits y PR reales.
- [Comparación detallada](COMPARACION_MODELOS.md): métricas por clase y matrices de confusión.
- Diapositivas en [PDF](PRESENTACION.pdf) y [HTML interactivo](PRESENTACION.html): presentación sin conexión; en HTML se navega con flechas.

## Integrar este cierre documental

Desde `clasificador-tickets-nlp`, revisar el diff y publicar únicamente los documentos:

```powershell
git switch docs/cierre-entrega
git diff --check
git add CONTRIBUCIONES.md README.md CLAUDE.md docs/ALCANCE.md docs/ARQUITECTURA.md docs/REQUISITOS.md docs/INFORME_TECNICO.md docs/ENTREGA.md docs/PRESENTACION.html docs/PRESENTACION.pdf
git commit -m "docs: completar contribuciones, informe y presentación de entrega"
git push -u origin docs/cierre-entrega
```

Crear el PR `docs/cierre-entrega` → `main` y solicitar la revisión de otro integrante, como exige la consigna. Tras publicar el commit/PR de este cierre, añadir su enlace al registro de contribuciones dentro del mismo PR. No afirmar que hubo revisión o aprobación hasta que conste realmente.

## Etiquetar después de integrar los documentos

Primero comprobar que el `main` publicado contiene el cierre documental y que la versión coordinada pasó el ensayo completo. El tag no debe apuntar a una rama con documentos sin commit ni a una versión anterior a su integración. Se etiqueta `origin/main`, no un `main` local que pudiera contener commits sin publicar.

Desde la carpeta que contiene los tres repositorios:

```powershell
$repos = @('clasificador-tickets-nlp', 'clasificador-backend', 'clasificador-frontend')
$finalCommits = @{}
foreach ($repo in $repos) {
    if (git -C $repo status --porcelain) { throw "Hay cambios sin guardar en $repo" }
    git -C $repo fetch origin
    if ($LASTEXITCODE -ne 0) { throw "Revisar la conexión de $repo" }
    $finalCommits[$repo] = git -C $repo rev-parse origin/main
    if ($LASTEXITCODE -ne 0) { throw "Falta origin/main en $repo" }
    $remoteTag = git -C $repo ls-remote --tags origin refs/tags/entrega-final
    if ($LASTEXITCODE -ne 0) { throw "No se pudieron verificar los tags de $repo" }
    if ($remoteTag -or (git -C $repo tag --list entrega-final)) { throw "entrega-final ya existe en $repo; verificarla con el equipo" }
    Write-Host "$repo : $($finalCommits[$repo])"
}
git -C clasificador-tickets-nlp cat-file -e origin/main:docs/ENTREGA.md
if ($LASTEXITCODE -ne 0) { throw 'El PR documental todavía no está integrado en main' }
```

Registrar los tres SHA mostrados en la entrega de la materia. Si todavía no existe `entrega-final`, crearla sin forzar ni sustituir una etiqueta previa:

```powershell
foreach ($repo in $repos) {
    git -C $repo tag -a entrega-final -m "Entrega final: clasificador de solicitudes de clientes" $finalCommits[$repo]
    if ($LASTEXITCODE -ne 0) { throw "La etiqueta ya existe o falló en $repo; revisar antes de continuar" }
}
foreach ($repo in $repos) {
    git -C $repo push origin refs/tags/entrega-final
    if ($LASTEXITCODE -ne 0) { throw "No se pudo publicar la etiqueta de $repo" }
}
```

No usar `--force` para mover una entrega publicada. Si ya existe una etiqueta, verificar con el equipo y el docente a qué versión debe corresponder. La preparación de estos documentos no crea ni publica automáticamente el tag final: primero debe integrarse el PR.

## Comprobación para la defensa

Clonar los tres repositorios desde `entrega-final`, conservar sus nombres y dejarlos como carpetas hermanas. Seguir los README para instalar dependencias, disponer de los artefactos y levantar API y frontend. Ensayar soporte, autenticación, tickets, laboratorio y métricas. Cada integrante debe preparar la explicación de sus commits y una modificación pequeña en vivo.

La consigna establece presentación el 3 de noviembre y último commit evaluable antes de las 23:59 del día anterior. Para 2026 corresponde al 2 de noviembre, 23:59. El tag identifica la versión acordada; no vuelve evaluable un commit fuera del plazo.
