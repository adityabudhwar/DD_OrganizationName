<html>
<body>

<?php 
ini_set('memory_limit', '1024M');
$searchquery = null;
$searchfilter = 'article';

$searchquery = $_GET["name"];

$pyscript = '/home/budhwar/CPE_466/Project3/MD_Search/search.py ';
$python = '/usr/local/bin/python3.5';

unset($output);
$cmd=$python.' '.$pyscript.' "'.$searchquery.'" '.$searchfilter.' 2>&1';
exec($cmd,$output,$status);
#$output = shell_exec($cmd);
#var_dump($output);
$counter = 0;
$data = (array)json_decode($output[0]);
print("<b> Exact Organization: </b>");
   print("<br/>");
foreach($data["exact_org"] as $item){
   print($item);
   print("<br/>");
}

print("<hr/>");
print("<b> Organization Chapters: </b>");
   print("<br/>");
foreach($data["org_chapters"] as $item){
   print($item);
   print("<br/>");
}
print("<hr/>");
print("<b> Similar Organizations: </b>");
   print("<br/>");
foreach($data["similar_orgs"] as $item){
   print($item);
   print("<br/>");
}
print("<hr/>");
print("<b> Abbreviation Match: </b>");
   print("<br/>");
foreach($data["abbr_match"] as $item){
   print($item);
   print("<br/>");
}
#var_dump($data["exact_org"]);
#while(count($output) != $counter){
#    print("<b> Eaxct Organization: </b>");
#    print_r(htmlspecialchars($output[$counter++], ENT_COMPAT,'ISO-8859-1', true));
#    print("<b> Chapter Organization: </b>");
#    print_r(htmlspecialchars($output[$counter++], ENT_COMPAT,'ISO-8859-1', true));
#    print("<b> Similar Organization: </b>");
#    print_r(htmlspecialchars($output[$counter++], ENT_COMPAT,'ISO-8859-1', true));
#    print("<b> Abbriviation Organization: </b>");
#    print_r(htmlspecialchars($output[$counter++], ENT_COMPAT,'ISO-8859-1', true));
#    echo "<br>";
#}
