/*****/ index

code_to_pdf
code_and_report_to_zip

/*****/
/*****/
/*****/
/*****/
/*****/
/*****/ code_to_pdf
use enscript
	sudo apt-get install enscript

example
goncalo@goncalo-Aspire-5820TG:~/Desktop/PRI-Proj_p1$ pwd
/home/goncalo/Desktop/PRI-Proj_p1
goncalo@goncalo-Aspire-5820TG:~/Desktop/PRI-Proj_p1$ enscript -E -q -Z -p  - -f Courier10 *.py | ps2pdf - /home/goncalo/Desktop/sourceCode.pdf

/*****/
/*****/
/*****/
/*****/
/*****/
/*****/ code_and_report_to_zip
example:
zip -r proj1_grupo12.zip coisotsdkvdskon/
