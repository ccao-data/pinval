# PINVAL

This repo stores the code necessary to generate experimental reports
explaining how the CCAO Data team's residential model valued any particular
single-family home or multifamily home with six or fewer units.

> [!WARNING]
> This project is an experimental work-in-progress and is not yet used in
> production. Reports generated using this code may not accurately reflect
> model behavior or CCAO policy.

## Developing

We currently support two ways of generating PINVAL reports: With
[Quarto](#quarto) or with [Hugo](#hugo).

### Quarto

This project expects that you have R and the [Quarto
CLI](https://quarto.org/docs/get-started/) installed on your machine.
A working installation of RStudio is recommended, but not required.

1. Ensure that renv is installed:

```r
install.packages("renv")
```

2. Create a renv environment and install R dependencies:

```r
renv::restore()
```

3. [Optional] If you would like to run the report for a specific PIN, year, or
   model run, adjust run parameters under the `params` attribute in the YAML
   front matter in `pinval.qmd`.

4. Make sure you are [authenticated with
   AWS](https://github.com/ccao-data/wiki/blob/master/How-To/Setup-the-AWS-Command-Line-Interface-and-Multi-factor-Authentication.md).

5. Render the `pinval.qmd` report using Quarto, either by clicking the "Render"
   button in the RStudio UI or by calling the Quarto CLI:

```
quarto render pinval.qmd --to html -o pinval.html
```

### Hugo

This project expects that you have the [Hugo CLI](https://gohugo.io/installation/)
installed on your machine.

> [!NOTE]
> While the Data team often does work on the server, you should do Hugo
> development on your laptop in order to run the development site. Hugo
> installation is easiest using WSL, where you can install it by running
> `sudo snap install hugo` and entering your WSL user password (most likely
> the first password you ever set on your laptop, if you haven't changed it).

1. Ensure that Hugo is installed correctly by running `hugo version`.
2. Navigate to the `hugo/` subdirectory and run the development server:

```
cd hugo
hugo serve
```

3. For a quick sample report, see the examples:
  - http://localhost:1313/example-single-card
  - http://localhost:1313/example-multi-card
