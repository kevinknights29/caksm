% This LaTeX class is intended to be used for M.Sc. theses
\PassOptionsToPackage{fit}{truncate}
\ProvidesClass{tcd-msc-hpc}

% Declare the options for the class.
% The class option 'lineno' turns on line numbering.
\DeclareOption{lineno}{\AtBeginDocument{\linenumbers}}

% Pass all non-implemented options to the base class scrreprt.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrreprt}}

% Declare options for chapter-like numbering
\newif\ifusechapters
\DeclareOption{chapter}{\usechapterstrue}
\DeclareOption{nochapter}{\usechaptersfalse}

% Default to nochapter
\ExecuteOptions{nochapter}

% Process all options
\ProcessOptions\relax

% Load the base class scrreprt with default documentclass option english.
% The implementation uses ideas from
% https://tex.stackexchange.com/questions/147243/a-class-with-default-language-settings.
\RequirePackage{etoolbox}
\preto\@classoptionslist{english,}
\LoadClass[a4paper,DIV=12,parskip=full,headings=optiontohead]{scrreprt}

% Load style package AFTER base class to avoid command redefinition conflicts
\RequirePackage{tcd-msc-hpc-style}

% Set up section numbering based on chapter option
\ifusechapters
    % If chapters are used, the default numbering in scrreprt will suffice
\else
    % If no chapters, adjust the numbering
    \KOMAoptions{chapterprefix=false}
    \renewcommand\thesection{\arabic{section}}
    \renewcommand\thesubsection{\thesection.\arabic{subsection}}
    \renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\fi

% Resolve the dependencies of this package.
%\RequirePackage{tcd-msc-hpc}

% Define some commands to manipulate this basic information.
% Defaults for optional fields
\gdef\hpcmsc@shorttitle{}
\gdef\hpcmsc@secondsupervisor{}
\gdef\hpcmsc@supervisornote{}

\newcommand{\thesis}[1]{\gdef\hpcmsc@thesis{#1}}
\newcommand{\shorttitle}[1]{\gdef\hpcmsc@shorttitle{#1}}
\newcommand{\school}[1]{\gdef\hpcmsc@school{#1}}
\newcommand{\degree}[1]{\gdef\hpcmsc@degree{#1}}
\newcommand{\studentID}[1]{\gdef\hpcmsc@studentID{#1}}
\newcommand{\submissiondate}[1]{\gdef\hpcmsc@submissiondate{#1}}
\newcommand{\firstsupervisor}[1]{\gdef\hpcmsc@firstsupervisor{#1}}
\newcommand{\secondsupervisor}[1]{\gdef\hpcmsc@secondsupervisor{#1}}
\newcommand{\supervisornote}[1]{\gdef\hpcmsc@supervisornote{#1}}
\newcommand{\signaturefile}[1]{\gdef\hpcmsc@signaturefile{#1}}
\newcommand{\signaturewidth}[1]{\gdef\hpcmsc@signaturewidth{#1}}
\newcommand{\codelink}[1]{\gdef\hpcmsc@codelink{#1}}

% Set up the appearance of author, date, title, and section headings etc.
\renewcommand*{\sectfont}{\color{TCDBlue}\sffamily}
\setkomafont{author}{\large\normalfont\sffamily}
\setkomafont{date}{\large\normalfont\sffamily}
\setkomafont{title}{\color{TCDBlue}\normalfont\sffamily\scshape}
\addtokomafont{disposition}{\scshape}
%\addtokomafont{subsubsection}{\scshape\itshape}

% Remove \@date and spacing following it from \@maketitle. The implementation
% uses ideas from https://tex.stackexchange.com/questions/288800.
\patchcmd{\@maketitle}{{\usekomafont{date}{\@date \par}}\vskip \z@ \@plus 1em}{}{}{}

%Change how equations are numbered to omit chapter numbering
\numberwithin{equation}{subsection}



\if@twoside
    \rfoot[]{\@author}
    \lfoot[]{\ifx\hpcmsc@shorttitle\empty{\@title}\else{\hpcmsc@shorttitle}\fi}
    \cfoot[page~\thepage~of~\pageref*{LastPage}]{page~\thepage~of~\pageref*{LastPage}}
\else
    \chead[]{\ifx\hpcmsc@shorttitle\empty{\@title}\else{\hpcmsc@shorttitle}\fi}
    \lfoot[]{\@title}
    \cfoot[page~\thepage~of~\pageref*{LastPage}]{page~\thepage~of~\pageref*{LastPage}}
\fi
\rfoot[]{}

% Configure the appearance of the title page.
\renewcommand{\maketitle}{%
        \begin{titlepage}
            \begin{figure}
            \centering
                \includegraphics[width=.8\textwidth]{Trinity_Main_Logo.jpg}
            \end{figure}
                \begin{center}
                        {\huge\@title{}}\\
                        \vspace{3cm}
                        {\LARGE In partial fulfillment of \hpcmsc@thesis{}}\\
                        {\LARGE in the \hpcmsc@school{}}\\
                \end{center}
                \vfill
                \begin{tabular}[b]{p{0.33\textwidth}l}
                        Name: & \@author\\%
                        Student ID: & \hpcmsc@studentID\\%
                        Supervision: & 
                        \hpcmsc@firstsupervisor\\%
                         &\hpcmsc@secondsupervisor\\%
                         &\hpcmsc@supervisornote\\%
                        Date of Submission: 
                        & \hpcmsc@submissiondate\\%
                        Project Code: 
                        & \url{\hpcmsc@codelink}
                \end{tabular}
        \end{titlepage}
}

% Configure the appearance of the plagiarism page.
\newcommand{\makeplagiarismpage}{%
    \begin{figure}
        \centering
        \includegraphics[width=.8\textwidth]{Trinity_Main_Logo.jpg}
    \end{figure}    
    {
        \centering
        {\LARGE\textbf{Declaration concerning plagiarism}}
        
        \vspace{4em}
        
        {\Large I hereby declare that this thesis is my own work where appropriate citations have been given and that it has not been submitted as an exercise for a degree at this or any other university.}

    }
    {
        \vspace{4em}
        
        {\Large I have read and I understand the plagiarism provisions in the General Regulations of the University Calendar for the current year, found at \url{http://www.tcd.ie/calendar}.}

    }
        \vspace{4em}
        \textbf{
        \begin{tabular}[b]{p{0.2\textwidth}l}
                Name: & \@author\\%
                \\
                Student ID: & \hpcmsc@studentID\\%
                \\
                \\
                Signature: & 
                \begin{overpic}[abs,unit=1mm,width=\hpcmsc@signaturewidth\textwidth]{\hpcmsc@signaturefile}
                    \put(0,0){\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots}
                \end{overpic}
                \\%
                \\
                Date: 
                & \hpcmsc@submissiondate
        \end{tabular}
        }
}
% Reset section counter at the beginning of the document
\AtBeginDocument{\setcounter{section}{1}}