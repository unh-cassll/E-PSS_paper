% Just some document style options that I've picked up over time
% Nathan Laxague 2021
%

\ProvidesClass{nlaxague_course_document_style}

\LoadClass[12pt,letterpaper]{article}

% Dependences
%A Few Useful Packages
\RequirePackage[frenchb]{babel}
\RequirePackage[utf8]{luainputenc}
%\RequirePackage{fontspec} 					                 % for loading fonts
\RequirePackage{url,parskip} 	    % other packages for formatting
\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{fullpage}
\RequirePackage[margin=1.0in]{geometry}
\RequirePackage{fontawesome}
\RequirePackage{titlesec}
\RequirePackage{array}
\RequirePackage{enumitem}
\RequirePackage{longtable} 	
\RequirePackage{etoolbox}
\RequirePackage{tikz}
\RequirePackage[skins]{tcolorbox}
\RequirePackage{multicol}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\RequirePackage{multicol}
\RequirePackage{makecell}

\definecolor{Navy}{RGB}{2,33,105}
\definecolor{Columbia}{RGB}{108,172,228}
\definecolor{Teal}{RGB}{0,128,128}
\definecolor{Black}{RGB}{0,0,0}

\hypersetup{
    colorlinks,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={Columbia!80!black}
}

\DeclareUnicodeCharacter{00E9}{\'{e}}
% Define default accent colors
\definecolor{basecolor}{HTML}{000066}       %BLUE

\ifundef{\@green} {
  \ifundef{\@red} {
    \ifundef{\@indigo}{
      \ifundef{\@orange} {}
      { \definecolor{basecolor}{HTML}{e68a00} }
    }
    { \definecolor{basecolor}{rgb}{0.2, 0.07, 0.48} }
  }
  { \definecolor{basecolor}{HTML}{b30000} }     %RED
}
{ \definecolor{basecolor}{rgb}{0.0, 0.62, 0.38} }

\definecolor{darkGrey}{HTML}{989898}

\colorlet{linkcolor}{Black}
\colorlet{accentcolor}{linkcolor!90}
\colorlet{symbolcolor}{linkcolor!85}

% Setup hyperref package, and colors for links
%\definecolor{linkcolour}{rgb}{0,0.2,0.6}
\hypersetup{breaklinks}

%Italian hyphenation for the word: ''corporations''
\hyphenation{im-pre-se}

% Setup fonts
%\defaultfontfeatures{Mapping=tex-text}
%

% Define font to use according to localFont option
%\ifundef{\@local}
%{
%  \setmainfont{Source Sans Pro Light}[
%    BoldFont = Source Sans Pro Semibold, 
%    ItalicFont= Source Sans Pro Light Italic,
%    SmallCapsFont = Source Sans Pro]
%}
%{
%  \setmainfont{SourceSansPro-Light}[
%    Path = fonts/, 
%    BoldFont = SourceSansPro-Semibold, 
%    ItalicFont = SourceSansPro-LightIt, 
%    SmallCapsFont = SourceSansPro-Regular]
%}

%New length definition
\newlength{\rightcolumnlength}
\setlength{\rightcolumnlength}{14.8cm}

\newlength{\leftcolumn}
\setlength{\leftcolumn}{2.5cm}



