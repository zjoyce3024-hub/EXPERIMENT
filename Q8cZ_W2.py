#!/usr/bin/env python3
"""Generate reviewer-ready strict-saddle results from five embedded trajectories.

This file is self-contained: it uses only the Python standard library and does
not read any external data files. By default, outputs are written beside this
script:

  saddle_results.pdf
  saddle_results_table.csv
  saddle_results_candidates.csv
  saddle_results.md

Use --export-trajectories to also reconstruct the five original CSV files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import zlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


_DATA_SHA256 = "68ca19324e4ff223be676a75c88850418a755be4d6727493232a138520a18c53"
_DATA_B85 = """
c-rl~+p-+Tm2LSidhJA*-!F~#{(|#(%wQ-Hs$v@n$N*Ays@wYCdyW<Ep1ChnWghNru7{E+iVS2%dUz~f_ciDL`<EY|fBNZ{FQ0z<
{`vp@^4H(KegDr-{`l*UfBfbD{L7CoUw*oL|NZOb%Rjz+{q56l|M~YXm!H3W{qp_Om;d$iZ=b$@`TXPC*UPuxf76v8zkT`p_(S{0
&tL!gAK$)z`sLe?AD4gs{`se$xx?3Q-+#US+xK68T)zJP>!<(x^!Lx-KmYngxA}4T@!RL$e*X6L^Y@>B`_E6m{m1t&KmOy}FF)C5
zx?fQTHw$B`10wOaQ&mpuRnkN^yTM&|MK-8pMUxN%jNUG{rL3k(l1)f|MTsK7X0JqpML-R%jN5rfB$SZ`s?rC|MB^^-@kwP^!wlc
uBHFmx8J}1v_8*+{J8!0Z=b*a_Tw`C;pP{v*XB>?p+Eon>*wp^+&+vR_457Ozy1D0JMv;Bm%sk<`Nxl+|Mv4=?VJDetH1b*)}}xE
H@oub+}evSyGuQ4zQn^t-%5<3hFo(`?cxsSTq`~25?YMCyX5o52cJt1=O250FSSx?E2%Z7Ta<kH;A1&j38kdsi}xkO%PW`c`yu<-
ysyRSCrVB8%Rc!SV(Q6-aEa~`^an>QEq4~@s{itTxbX43wr<NcJxy;p$C^_Qv48pC@<)H%{){eZJ^FjO{EzQ?v-Y<izkUDtufKg7
a(r&=xoxLB{_^eH-#`8B=dXJ4fBp9LZ$E$k^~+D6eti4!cYVT-ElgT>Zu-?0d+1U3w?~X@VA0N|+IrK*{ntDA_4C&!|Ks^DW6l4U
1mny2``2$@ufp*M%l-B9|N3-2Q$KVk-#ZI#uJ_zhu0843FZl>fdmp{;T5fXbTweC6hT^p4y=mJwFFuaN$C^@UF_f&=?_$>KO9`P$
I#@*u=kg?Gy=Gr?jJ3wFdGTp1UdyaeyWV5<F*aREvFLEspckU0oN^O%0CWny9<})t^5(_ovG}H)mre*V_^LhSie5zuB~>Yip6a>y
ka}xfpYX24wp-?lExsI*;^>0*zPRRdy7VKLn51U1%ehPMor|w2G+onU>Ek7&&8x3t>z%eaNo2e<i3DG(I)WZ1)>2q*@8|lW9;PLy
Cz9w(jCM;-8=IfH<e^vTd<nL`S)_1!^14`La>5XM=_22Ca*OPcglG4e?PKxIOWVjA#^ReVoa7^T*+VDkX|4=~le6nZ=E=M2TBv37
;$@gveLiG@y2nQLl(ODOJ$&j?=-r3Vc(iltyBq+W2MK*I*y7Er_qKXjRqbbxDv=(P71GswRGIf&a*ENVd@eqRB#qju908a8ZbRk6
SiF~Z^pXb2NK3V{_#zo<I-B0f`nIurZGM+Z<pKHs-OG<-`BIpn;=g!=S|Xt*@}+AbHAP(>t9M#fL~pv9wYuG_KP_!Mq!SV8+#1?F
vib65q}R1jxwz@{agsEsw%z04RQYMUSD(k~eTggshpfw`MBgWzT`e_f5iW(H@8#L*lX8E&WNN$Ul(GC8Wm9FH0%}nXxeoMEgY>(c
9bME580(MH_4<q*=f2<E`Z^v!ny@yxVcEHU8eD(m8rl9FwOWqp`2f<ZrN}(XyVs|;zkxOdNQ-G9HSDeA#tv`Qfe2ZmpS|A2IK4$t
fFdRt=O*WWx5ME3SidZjoEo2`(5nyq65P?{L`a16KRsGK?UsVN(DXPu+Ma*j9bmaBKk{E(4!uaLdxapZUuRmeRghQ(*V`#Sxwh!!
Imid{*stB|KkY8crpv$S3dV{4iupV-$ko^Bm4hseboxA=W1hTTfWoiss}E!K^22HgGQUp7R!&0b3VEA+{1}5ayPm6WQaVY2dDDB*
v+UMA;k3K{&?#+%t~#quwm0dm9Gz9Vs233X$e-xn<<{wXmhThHMr>Vv8q3#jwS+w)SS(%3FNXrIQ1IZiz&6|-c?&XcCCdBHI_!D3
>Ir#BPYCiCoX)C(+8_zJM6Z=JVn-QVxozGSsDJSUG|71^#P;QvvHyyps@AEqDb?dkm0%qi1Ld`Td2V_Mr~K5UE7Cc#!22^(b*x@t
p^Ox{0||S=o<@@%IhalFL^||5ys=u0NQ|q{An$hE!fAKik-gR>KRCyxXk98Xx<h6xD!7u;k3N)hh4PtY$@HB-VBEJWpFS18XIa=)
t%i89D+ZGvA)_m26Ja?ue<5|)V<X=x6R<ls=!W=2xnVk$2!;H7Jy6yg!HK}k${yrXcUHX`d2=c1h{*VEPi^|Ke1(f7T;;Td5G@=j
M<bumqhc1@X!F`yl+ui5wq)jZ=Ne)d!k>avqN`4!S0>dTahtUYtuD(!?kS9PgOp~_lj?Ii6~68EnqwT&kCyLp5OQZ<TS9kg+f9?o
`)QJ+p*urJIw!f3f%d!I&X~pvXp$W{HCfA3NSivY1L*}$szTqMMxL<f{Zv90e->MIuRjm@53)zpA4+IdahByd6J-k0_&Tk*5BWhk
QL|9&m5bY+uZq+7`XU|KoV1Oz+?uRcOIZP8msJjOl+WdNsU014S%j9_ZqFf3+pC`n7PRQ_CeOOGdTF8*y)<erid@F>6#-Ogg_Huq
cJ<T7`bmTLrZZXPyk+voT+2nCn{TqpwdN7DW89+ro=BQJZGQoMK7b^ITwy~o;uB=WE!m?XrG6rYYUDd>z!vM#6cg|E-I5#HM+#kj
bMBVkX$o9?J6!29P%>()H=g@1cRx^-^Qh1mcOTy5$NHuD<<^ic&k7aM>WQqCmsXOt><V#D-&5Wwk-ZFp(<is*d{UU+Lk>EdR8}a|
nNueD*S-fC0J%Oc4BuIoFv^PNsf})rd{dm}t1BL_Ri;Y5P1LW}BTMNh!jj)vD6fq2XiYe%XxJFG>z}l&Z>a?3@;0Nqh<xdX!YxH$
IsiQ&G^505o?NrkxqPDS%g<x^F=AvDmk_N~OL;2l(pe?ptk*K)M!5@7TW=L?jKl8bm$7_rJs$FAob-yEesi9ke!eO^PaM}#<4y7+
v^c!%%D?P>fI9cT@Lz3rc1@A5o=f3mCv=gR>jj`G?FBemAbfYau26Zq`k$e%+y?m#3bPck=nQ1`zZ5!(iuPVH<%k_gt?N8=%bLp%
+P(fh1xQY(RC<f_7=<dgq8!;JshB8l(-EtmD<tJ6AFn03+KLx;d(YWjUjVkL1x3FSZW=VCB~vyBRK|JKyf`3T=bAj7Q^c@)`7`*H
>ah?-8l4syx{^BmI~)m85^;7VRmqp92$Q3{>fOtqN#|(EohrMlZH?#VD?seH2-5Q|&&}7dbyToD)v(>_Gsh)73foI3PolV;v`KR$
Ym>DF^s)CX6bTOpTn^}V)0NXYy3`+%<N<rDxP;x89qWPKo<fCa?0%p#NeWh1-S)n!X?fA&Qs=2rkWg5T<0;=VCCN_kis?t%N>)M9
e2__VRbSaI`en%uU2Lh6d64Ip=+e`>z_*w1c<G)Y{9YUH^|X3Oto8N>s4HumBCSb@#SuB?BWZ`V{3>1A+FYR4O4wzo^u9fA%q!_E
r&Ef>b=S~AQHIOQ6(U7Km7)y0RaqOkH-bsO+i%D->a9-BjT8+Tk|=ACaXV9%H{E4v-SFC^_jNh9TogFc2DiJfa9Z6Oyw<L8y=!0P
>&Weo_{{QwD6TtqE>QN@wXDcK%5^ApyY4JA>Xj``iVT|zvEk#$%Zx`aNdXsWHm8v6Ogm`Twex&HCa&l01-r7+&ZIkKrgBemG`pAj
p@oo)f1$=0<Ip}N6_b2^xp4by?PXeCT>CD=<#Q-{4;k%4icmVn(@vh^SfLJu{uw1J?OC?hyh>URAUBOBtD2~ZO0CC39<&18F7+9z
8(Dw-Q6lHo70we7zq1B<b?AXQZR7Wfvw8}C7|r6znl`FTu&2#ds(-Xtxn5ea;%}2*k?vXZ2L(Y#pe=$lwcPA}d3i<d;vq{SaYcCt
SL5YF)xoPW`Q*PF&8B1gL6;-ZYOuoYIsT9im#=rU@dquX6T(o^>d`)?hw|57u}{6-_yc>{wIN5ReN-ltBw3ViDA_&~8s#jVE|Co_
TjaqLRsJ3b6m#0``<9utPfpUC(9iU1@>pK0rMgUfhrJo?)pCI4L#CL6HnnWe+?TrQh91+eEiW>Sj^G1FQDKOaJ5a+YTLmxYtZL&4
VQFmJz5bOt=q+fcb^7$%^hV)-$=K6!A=#~A%+tv6^@^o*f<1oab^)5H10BRh(Yb3qI=GqO)$;RFayqkwv12MlyN>E^y;Ldt?c)V2
b?=KcYii9QN+r^I-N>*qt0Jw^hZB~lo}lD2?t9)XK-I6zvG$YxIz@p}IEnUUqVHS?XmcEOPrV~lrqn&nNbS7~pRZaV1*aq}4yI8Y
{ZNv7dfNGIPHb%Y`C!L+7@&sGXpM~G-B<1Z(DD)aPh-eS{${(pyrLN2TI(sdlA*3LKy-WLYy&;?uIFKZ+L5ncX4cMs)HQ@4XSyj;
<=WFba>;!RdMJu=dKG#T_NlisK-x-cdZtEC#+}Fp52{_FR%eN9MmI7;GG>4jA$6)B9xtkG@B656Rh}BPB>u)6!=*{1Se{)<&^&F^
-dZ`Mn5h)4Z+owLO{-r~a#H9Ib@H$E8noe~QtU~G$r+Ie&6K-$2W>Skg&of-HsgaaVAcb(Q;doZQ=@3J`outFBvZY6<*UrmrG!gh
<TT0JQ?XY}$xe@{T0;CDn5}BG?CxIoOtDDgY8`=TW5tyWpnC1D9O|i=gBf)ho8I%gr*&V~%<Yn_4pU8}(Z_PkB^TOH=69z$_StD2
5|w&i<Kc(a4q`l=UFlCL$8vdj#T{g`8yVM#MM&Ph7jU7axF}@F3!Un?-F$nhV=c)!#$M$M4t0EU$E<DfALZ9(8q!fAm2bMYRL70+
8+RzS9y{2qYn)zRIX>acLwV-pkWz9axoyv4)_zSR6q8WB8&Cd*F)hU~v+682$Se#~W_5eVY+Ly!1*oS2Mric1_3z!SoS%G8?Kce!
LE3qDr&^m18+|SLRAUINR6<3~dN<XWBG`_%maNLTeYa<iMWQ{^l!Hzd(jiG8I3L?>GY1Ka5#`h64N8orGwU&?7JKU31WtmyGDUxc
Pi5#ag<I#?yyC<AlGz`!+&SkO8YEPwcXN4pB}tLd0wJ|#v&il8Ayv9S6&Sf7x0d0xQ;+2%+<f`eW9iVs{5jj~lj*UicQm)M8cu{(
F1z91T92`XW={2!r;r9crOuGvz23Tnbhd$>Z(Bw}*Z14nHWPAT9xevQd(kVF1Kyq7;aLTvQHAnSO#4wO+U))u0;|}+%Q=^ipzvZ%
+R<xEDwNunH{6HKt1c^aj%|1hN!}UFs<n9%akF25s7~iB*(_(;-fq=q)@WF1okDGLph;VC5pV({)@O){2Khu6$M$MhYs&*J!7LoQ
t!j;wyjo@%!*{3uHc5%JZ2y;bs9|j|@#Uf28_uN6o8q8IKVcq|>(U#+Mxz4#VfO*-WIZ%<V9{Cgv)<M#m)t=ody5?WsQu7V^v!TE
6HNu#lNBxtT}O&8lBw?sA11mE>HKleI?~HDqDHAa@&%@b@$c5Juvvd9x_|M;`?WI)!;$HTVq^?_3%O+lED2L+6iJvAm75zw*`1Z@
YZ}FyI@8jae5+uJ#5BQm<V7fq&|2i?q_#Vn?|$_njI)*Hs@o_BvXWhyS|Q@^$oLipf3~9)^y!#+^Cczf4sAvmM_$=*$`lDR%-X0w
#IoDo^*QJnM!b=ZLQ{vs<pGjoG~1A(0$8Uq=9AyM&Gl6+MuOUwLp#BU-Z}7|^H^<`c_t|t%jQ5T(j$9#SDCCFoMRv5L)*O5<>eJA
oy0@}ZNWJN`QG|oI7-x^sg%;dQJQH=ZRE|@r$3nkld=YiW2wt$_cTRjp|TuPnTu*F@Gbqxo@R^Kd9oPlV%CD`aIcVUq;BLOcRh1y
Htf5-FSXBkSc`WmlOQs+Bp#A{n^95^v}rq%BbXGwU5hQJOnLzAdg`%axoc~JtTVt&Iry%aDAQ1zwm0$jnJ1%tZLT{qHKFrF54n$1
0jJPFE0Ng9vDglh)NZrDI)F5|qsjs<!PI2FYx^ZFL0(4QUbgGA^GIeP4!e=TWGi+oYo7fsog=En<1E#>BPHpjh-~JKI)<%PKS}{H
$)XVnbr+`smf8s5F(emz_jiT!iL|WECu9q(E-$ZyBPWHyc+;NxL?v!#s?%^>3RS>sG%tl*f|pWt_0+zK<*xTDrhS_f^4ZPBxg15J
+uwL&?aLN=?NT^KEMJdG=P;wQ?gXB+k>H;36J$U1-u-(9X@9xAf|(lr8#DD^2s1V4P5uF9DyDEnbRY?m^CPudS6Z!57@+KVS3DQW
p#WBz#vmGGxGt|;vaz|?6-Zc6<^Wxs-wh1h>8R2dwjw%RriiQ{siqmxac$gnv;rMYYEKFy0FwncDvr?2uRMU4%9j1t@IC|1&<<iz
EEg*Jm0rU-&Y`bBQpCp8h-FdV?;zf_<cXQ8x0*?$);OkHbcrd=YVRDpmf7Q+7(s1`32Y!f_bxw9_)8q+DuH3V!dZpclV>eVp=s|&
je|5iQ&dui;Cz<Nt50k7MVn1kpn^g~9zb%NE0xd2AC9qb7OvGPJgwE2_pEl@wfsDmUuYea`1GDZ2eG+Q8sLMBHW`*az~~qz(x1y!
Qkc4Z^CxnubX+99lU~s3fT*j7Bp)bPa`Iq^jj`TH&#*L6lxYop^YZHez?2{q>v8rJb~{7Ko&nfqNW4t9pR1R)1YVk@=lZVc&byZ1
#_~gzDruA<qmDpeCR}RWIW0w6a;Ef?&MII~&C}ZBc8g9Q5Rwse2~fKgs+^h11Mk%84-%j&0;aF@Fo5LX-xWvfJ}Wg*+&Z;)ep6xr
8n(^lXqTy=6rVw$7;`!Fy^?!4GoicJ?<Wjnmwe;`;vbDH;5kU#a|H5+G0i*%N-NbX30joo`1+pl*?sE|WBtkCj0+(;f!K+9+?n5{
8(OO=6ZO~tVwp*6-S0lR13)#>R|JkpbBu=gaL(g<*{jMpKB=EEQJt-6N(Rd@xV+nC@H1Vd*iyT$qsw#HKr&cd4VVNDFH8Xwiz_*1
jXbrY5Vk+S%v>?MXowELob_TLurX<gnY0zf>7hOKPm1-K!D#yD-Ii;>s7hEeV9XI*cwc(5^#iCO?@ULdhBD?drTu*ekS-X7!*1<A
P*j7W3%X^ZPUVSqFZqwwY1(kxT2E8{6+~4FbSi24-OHaTt-y>QK#Zlj5Dbbgg&^-AWWWZ{(z$#}rj*neyXa+iz~a|jU7`a}4jb(W
#b5eSC_2$=pYak#Ge9MpP2d6ooltjc&oKL%8j)3?tZ=63w0<0W1=}=frm1|WLAb)y4sNR7t$V_pW>*ZzC`QXR!%Gc1a&{|;&~Dmi
Do>kw@Sr|WA>O_IIaCK6b_Sg)1Ym;=IbMPe#|A*uVRXf1=Ic7BV}-N3-ECeds?zObNvW<#S}Lv^Ntt9L$qE?5@Z2JR4^sUZpJ?~u
7ltY(RgN&lAsJMTE(AD=fE!&ri>#v1Zmoc;fbgGo>-{jht7+g3U|Ep~^Hw7!;DxgV9GRX5mP0JXf&(DyV#&Mh?t)Ow3?0EcK%Z=+
_YY^t75SKeuFQebSb8_4E^vQB*=@HLhH7R`w?&&)Ck=9G2jkW98bYV=RfpyW#YF#V&?epQPL?bPRXO&>AY&@cbtm?E$YA4m@RZ~v
4D5B;WZH_E#}Bc%?W%t@5(9=%<m0<W=Y%0aby=xeGfY=*=$_CNmIXOKm8p-oTl2>SBjfQ8LYMSlyT}zb#7R5f6sM%I{R#>?z_tp4
GtF+hm%loTGwC+Qrxc8k!fXc+T#_aEZqlB9wsV_qa=_4q-gd{l@!H1{QIcpCNqjGMcxgLT+^!gYQj0OKs`b-lT$oGUt@~rVLTMm5
fCD>@f!6PJ4CLZ)osqneX-=g9J7AKP%J%g?F}=(sFk~v((E6FYk`tYn_?XNUA%gu)4Az=W+-_cezB>Jt_*UYMr<_QY%ZJ8iRxp`b
n97rhGYOBUcRjr8EcSg5P_7`73c{sZs?GRjlxJqNEIH7XLY`PM3KScCX;hr$7HwCbb+xAg+b{cNDIRAeZ?Xk|9>^wSKOKK*qg;Cf
=^T#0-ATi^+H3Ll6%Nwih5taQsJ05ilHAWc0oViMl}+ZkG6?^!2<G>#f5EtQcw0z>>&()dga(xpXPBgtr#>dm0?qF%u13e<*eyT{
M%572sjWyongL71!X8&2kflERv++CSAVEV0V(;$dPx}k7LkIp_My%2(#<ZHXFvd#T%Jd!*;4&cfHro2PFMr0pKs?}E$|*LKY_QNe
oTNe4AVfb=Bsm>b5rJNm3|Z^jA7Fu~HXH1bKxYChgF@>mGJmVp1LZ%4K!C#G2+;i9w!5DTMOCq2$OaulJ5U8mL01ZFjaC;OxGY+$
POXhVI;%15UjGb!JrXS~NheuKvcTo2)ZE$flwur7e+K$p10v0xW1G9ze+6^%8Tpq9?)XC1=nxF`FH&7Ee)Tz2`Zp-Qgt9yMN_Anm
007}ZRDf;?^Ch>9Vpk_5tCOWv=9wvEh;U_++jjAv<~Jl%JgwnD|0vxe^>1_vsT=5enr)yCv_?1;c)aa3mxZFrU@xP48Hz*N2U-NW
KLS|WS{cf-Th@o9e}YnQcb~=zeSv6WoGXKQBHhV!Dy<v9Eb0HkfMTGm!7Kq?19~yrE7o}$Us53zy$GUSG$^!&o}sfx!Tj_c0)V`c
Sv81Uw(Fj}g5Sz;z5HS=ittyye*)C4`OZXLEE7Uop_eOkDitV1wy%HMUJQ|9SqVzvyxLecc(Al8;a1l{0O33NLfJFp=u3IG@wx$0
^$h&f3>;Q8ASp00j0Kbi*)gg>a@Sc`M<?%Jc02zINtNb?oT|(h9AwNVAM(=}3e$cE009RMQ>@vBV6*kx?_T}__W)Rc_G41w0AOC=
5207aw+8*vSiUFa1IBYV=+)i6=fY8?6S4q^*<(&ppkOH!$Y(k+V)1bf>=AkCeRN-1*xxX-fK-jT%R@tuz^)^9q20t;CL(!1Gq=*E
u$$sKGr+$6`1ZB`_+{Rx0YR>*=6o8yU}@cW`KI#;j%Oor-hl_--D<dEU%)IVxsa&J4Unn;rUe7mo)IaxhLa0a?z)=8yX&9VktE?U
KOO!88cU;@lbFOtG~^S@69Qj-W^9?zNN^t7Y(7)g_jHFR+*fCF@W6lZt(TYNmK@q3D7|5bCzs~x5hIy-CY*46S7$inVdL9zc813c
?~K;i&TH0ta=oK{EF5%az)~u)ch8|D$KJit<p1^immlAL`Nx-^{y-L<gGy>!t2OT!%v^^gPbRe1OH79Wi~>yFKx{e#S!o=Oq7))1
61&R~Ww|moBnf_rUOLNxfVv{p#oE{0z!OS(8!-gXGac;UyLCy4*F?A>%cKf~q_;wZ9AFPBe5WCDrO{f?bPooC!|krjLZvOaqb>+B
WMnK4B1a1bZTQO-aL+gvdJTCuO%^Nn-L62%*DxGrY)8KWGp%?skY_g<ff=zHjtpb4XDkjR)$Ss0Dc3RbP$>E|uzqW>PG+ll1_SYS
6VcPd))M65=#TpLGDKO3@G|UutR2iB;MT~YWqg6wCJC>iEqTl|idl(edlq&fs!AI|k}Veu;HpFxAwc#7s6)E8FtFO>Dz1K4NSk*@
;#WjfaHN^FfjT3y{npKpi$4SOIwwmVpgdj{!oy{_d*)#IF)&Lf28>g0F&lV>bJbhQ&r^sSj=U!~)j0)`zUB>nO!b#@$ss_l&Rj}L
#oDi7$EG&!B{EX<6aTJu<0lH%(R2Td8-6HUWC-Tvl}lN^YVT;~Y*Rd!4@KdBA<Cg1PS5CaV(4$ljt1WRWV~(MFbu#Fl2I+n^;NK8
1F88a*WaiQz>(hLx{wBPhWpWhGHMO1C~0S&3`6Rup^;WH-6!|U(~&@TTCYCrHk@^NE;8+{6$YztQ^}w*<tqn;qCIn0!|;0$nge4T
q5JJeZ{=X*a?$M#_j}hvP(Y|Ox8@UU%!Oq->3gM&-~r3tz5WH>Rgn3nBSCqC&QVw(3so|?Cq2i~=lZ)%V9@}g&~Cf@OwXk;83r_)
gdOw(bhn37H-a7$W1nc=0R*2&rQCBMcia92u}aXPxCzcsW~wS#U}+y+!1P2$e<nyF!)`^7nG5d14(%;fYEx8jH3g<5btDb?2K~*1
zKpsagp%^q-^Id4JeG3TeCmf@v;oq>oF1qJ`sL-7a?l1zp{24jy77nB4MCNz_=qf|p}y&6U0*IIP(YV^2(=VN?|j1kJ-y=q3TW0D
1!a9|Z>_%Z{W7o+He*4z6|FJwp@f4f1;h+A0?n<YN+T;zS{m-e-QwG>YzEq1=i#GCFkiJJ2E~?o7sPRXkcjIQ75Z!B-!gK#-~RN4
<=*+l2vixgE)39y02-Z?iA|4;;|G~eHS|5`Cw50}?TV!@y}=|VB$Fhlq(1-zkf$2ynw%AQ(2f;62X_KL<?Ro!Mi)F=T2Ob{EMq!O
*q`W0c+MPpl*Ykn3?7!3{rex50acWSZeu_Y*ib5{q!0=J0o7VnJW!xw0(Dj{uiIV*Xz_}0m}peW6x@5X=#5;tJQf*D#XJ*OM`t;6
0D^Q=+26$479>2SF1c57PUz$#d<%sZbgi`NEYI)_*h6PvHgcE4d#j+=ff}<_kB*Y)jsd4g8``1Xq3t9wHrS+bX4jsAO=oTBq$It+
tLgtkf!C)OI_e-N1ajt;OW}yXPDjhR(Ok+mwXkAnPf-!E=j@#}v{0B{-#=|AHW^55Jx3f|3s3K9HZ;MONJ@j=`t$(U&V3><8&Lz`
D4qf#Av#TvS&C0OktRq=DkJ*_9A!=2ysFZ6B=Q-A13tUb(#6p;IAq<<sEwraq$N@ZJ}|pzzmgAH^kiqLA<}32<pVS0cOc#!F2QJu
Zwa<=)$LGEuU*w|7uc<nqALjBb+OtVLWTNQr$)|Qw2w~oJIo_QZdrHY4?aHoLHlqxAO=b#J4<lf_2ZMDiuOZq=T_0$=PZ9i5iQWt
g4=Qa%F&@{i%hA<O4Mmf%+|-yD03#7;aSpMU^;d};TsDMCOG2o(ed!|YSGqE{?!-d3hDKp8$3P?eux|l${!5Wo;wVwD1*WU967cw
edOBBjk#u%qp6U<;t<#lKl!o#;10howNjy|)c}_=eC|;`hKqAs0V1yFKq?>%b6brPFG7+M<vKmKttA;L2@_knjz*rBS2-Jr!4UET
9jj|6Y2AY!0{sB!EIF)wEezrXA9<X)uQzu-_f;`MXINWKQ!+`ZAD{l(d1l{Eax0R{LW~$mJDEXF`e3MN3AWZw@U0``+j+L&{0%TY
CTrA{>3(F3>AO&z!5tj|<t+9;ziL!+Q0sN7(!s~8<dVFTiuGWa6%6W1m@!C8nIM^<$B%Yx-ME}cPlp61cte4Rb!_nMZ@r1&Y8#9o
mT97jPb?uRV^^b%<KPo_%<#FZ<!DGv=JRmlJtVi@(K|sx0(fAl!OQDcA-ktTzfwQ-I;=OFst+@!m>uHR!rIiEo20p^pfySx6y{>b
rrzFo=vxSpm{xWQ)ZSX(Hl!AOCuxRkL)6;LHWWjmZ<h8unv@hpPXh!|W=0Ue8F)-F#Ga@Xf{%~+9EdcpT$bknlRfbHkfo(^qZy3N
q$ptZw7D$J09lOGYa?$yH3RUi>&{_n+~DQlG=7TJw~tSLtURb0{s_|BLGJ3!@6z5v*&OD+$|turS*a|oWNs^v22mpJfOdLlS0Rq2
gQ@7Pv@k;6msdTwtcj6Y$FQ@v1(k6RnizF@36cpv>jG-zE({bC5uKZQUYmEeO--OO%1fZvF0?7Q*Izq-=?w+9Be<Xxm@y^Audu9+
4ffUgjU6Pfa3GzI&%ubCj0V_vfsK4<kM#^zfPrsQ`!P(~i>vtH87BA0NN0@k{FR$~$X}4oXNaUOXYBSa=N>2PWvS2~b+gpz*{!+l
N6H64@@daYb`S3H>nP671UFRUlXacy-|3gIuh0;7DOkEKl8bX+k(fq53sWd&Z6DAGK!gLC0WC7E*q1-JoajcW^ad*k00{ra2a63j
D_U7OS?;P4ZtpTRg2d9(o&^i`5Z>E658-9xLH-<|hi~mGfcfqWc7WeVX2k$GJ>TD<K)x~3??5_nQ+TU&Qxv^{4zg-~Y?BRb8zNP`
SL0&*gUj)to2r>_4_#QkkzV_>O*(3;6`9Ai%LV##shFZ*eeUwjtxvrh$O0#(P@#FUU4Hs&=M()&@l~kE&e9g&;5Tpx<=9wA9<gbP
k(tTKV`sHz$d3N9-o3{*R%WCyFdsM+gS+_U)p~g5KS6UM{hOc2u53gG1yZqEfkAU^s|VUU15;=x(6;sV&bF;Oep2<VGw$fCnX@Oq
mfqN904>;A<*zUal#HxPMtBM@ker~cqX=ppTiogJ#~tc&`|E>>SE5GQ7n&a!-iLyCdD$PC7zV~&^JD65l%|h^r=H2$7|Z9<y%laT
cqv`UQ}^E9dxR4nQH54Hj&S1X&%K4W%ee44)!}g>alZrb*g86LX;Vr@2LiC-IKS3(bpp|2pzq2rDNa6gE){sdg3*nY?&T$8U~VxR
ODdtQ_vcrEfQ48RAF%kVcS~eteY&yD8oP_@re)OFt)mf`t<omP-0iREQg};I1^n|I6V8!t9O^PQe@$gS;p#HZMuBO_RHHxBn65yH
yKC0ugWNb>JsaO(Vl~Xp7nd9yv(ogD>A;dBN`8FTYO{>+w{i^^lI#M%!C!Mj>GPg9wGC~D@$Le$)~mCqA{@8B^5BNPE|t+#s$`Px
N>F|j+LU59Rpd}WCWS~t2YXy4Gy~xZh6)7`vTctg4@T7QQ6zCRuf4oV#|8t^z#|2i!fta1oEOU{T4r&j4sP!;QwIj1V7!Ap4~=?z
W7DYc*FudIY>N4<HEKLHYNSF88HoaSIRJfVRC606txdEq$2t;o4a%#4o&#T?r3alZaGV|5+ZcG$<qs~?1A~1o6pZDw=5p&0G$I*B
gQ`L3UMc8r@3Y)4I&ezzaIV-sKe=}|e>OsFSy9uZ&fX#r1NjTn-wB|#*xH%OU0ciKG&<CXj@sN>UF}$252IiY54u`RS~Gb~j%se~
vx`~~scbF9xU_x9w+=y(<I?3}McUY|nECRln5eAu)S7KO{j8fi4^u^b2g(vqO-7h-`*UwCzUiNcZ$GI(WX#CWotgjz(X0&{=UgsM
cu;zECz*2(;{}up@$&jrsUcVZDa{vS3A|Add^j048Ja5fYb0hvwvB;@vLdmY8;^j2S`;c=9^>iCKK-#bk=%UhUD^dKDNjDQ)uDIs
Od{DCGWeyAZH1%6Mjhb8k^AzP<R%KAjB@e6X%k;u)fsK|wATQQ`KK`gpH5)RL@>R=)tODro})rfGG+Jn)+5A)c7;)YH?pT6pZr{X
Fh=+<Wj7Dm1@n^`+HyL9rW`V4xW<o+J2|5T9nThMK<4S)qgxyquaPfeqMs(S+sn%olRDt>5E;_)7;hB>KTLwTHiDUJ9Cmw~xyO;#
9l75Wt;ZhU+<NXY?eDP5>w|Anap=W_7YfD`j6G$%$hoy-g`hYzjI<iY)_REJX<`Rcq@{;LTo7w?o;yA813KcB%k=Ctlsh!idXX{T
rWT>AOZ1X3mmk7Sh<m-wHN<5cpog2Sy}h-04tm%^4{U;ZyS{N2te1bPdL_WpX2b|`bj=W>BPy)fpabTdAI^eB>mfw-RP(tmuU}0%
CV`17=5Q5p-6p~8!+{cYb=01(DuVKiLNlE5r6O+cJasSmR8MeZDx!aU`fG2JMyRLwGH#%afl?a~<K;F;P_3H}DDSx324+JF^XeY6
@}NT&A)TX1K<}^Rl?8i#IU0zpej%ftP9<5^O{gUils032zd8;#w^@z@8ShGyg|x%)dv9;eeLotFV}Q=01oMO5G!0h&iS#P6$ObJ`
$>jRJB8viG7)V-R0+h#mGv+g(6b4O6?$WSuFRq4zJVkaHz*z8;Z=GH~?LmwqU!ZLsdmN4v^(b2R(7DH4yWZLMSm6epQ1mp^cKW<e
e$Bs$^tQ|EMy(xQ0Xiw;Vnmp2ZYmkMXuP?ty9f7F4v)tdG8DII(Oz8jinEf3q?>v;L*qAstHTRiuii70{9q-|O})L%+*GMz;A0_w
nO?)~y-jf84$ziH+=(3D+G}Y4L~t1*Z_Ia+wDxDLgVHEunDycVa+CUF9xtgC2L5PFFp&PsA6#bgC=w_(*fqj!)r=3rx5W#od}ZYt
ZtgN9*RDk(J}tV>B>Byq&9=dkMe~2a#>6*~+<r<f<AkXKUmKu%9~*1)V8-;reE!<lnss_$qb+V!njYEM6Z(ieL5dUO{gul)rJ3;y
1zN$-*=`*&&r~D0_fib!7{S`woBOP-g;hz9*VB3J*}X?{F#*=d?7pjSu6X+?zpsWCbO8a;+#m&}J7H|IFY+ObI*P`C{Frm*twox;
W+$c|17)vVu7@-$5ZKHGK)ZIk%T8hj>l=KsHHB-k%ws5t>00S?mv3)9>kl$)HOlwQqeS`i*WSYKg+T5PJ%_}&18@P|;P`!+Idjn3
0_uv<(#IS!Z$TLSo&i|!J_Y#2<wmP+ri^P$z3DbN4rwnFq|ACtbM<mV4F)h0DN4U~_U6`SWv6HxtIa3r{-^xj{+fU5u7#7O?~n8Z
4q^eWi423Rb(_^6eIm)~d2%7z>IVh5oDizGuMh%V{@^mm;lQ#6+oLCqV)`(eO75+<8J4T~f(z%MP3}wo-rjg@azpRX(cayeb9nMw
kEC^8WfhG7%_{gW#41?sAtybX@Nt!qJ1`^)w5gof+4NnEq0a%1m%@8?t>3laNsuRH$LtQB3<_Fk`L4z{2w{e_%0S)pG?;+QPbp0L
scc?-9IJ2C;wjx1$bunl2U*`*D(iJzm+r{~iaO+_HSgZ~H0cyU*UaSZGf)81E7q@EZ6y?w=D>`+MOPHfTb%u#P!o48KVKyWD0G@X
X&L2h6trMaTovjZwKtSvqlkrRy1(bR;9aXPlNge*ZZNnFt(dxxj%^}Xpi(B&G;lH$K+poMHG0xcn^#{4xgUToOk$GsE=GWHbY`sJ
4W(E<hGl8ODYC%512eH(aN1b>bGi40t^_bCWt2fkAH3?t%Wnw4bmyMp=Ee8PODLBi&=YQGmDP9&1Kk~xl<Jw49Rw?6_NesW_3d8#
pvEH<ui&({2#<X;b{O&ujE$v2!{AuFqY}nEACok@<z^7$rGu$numT9!I+Qn+YGMKYNYWh#Jqc+yy85D=G420uZ9i!7lI=YuB10sr
#!Drw2ziqPyi&X$yi~@&)v4h-*LUm7L5s&ETcTo%mm3YY+LQhP2#M5jnHZrm!#dcBniFloez(RPw0Q9VamO{_Nt_l}SwHno#@lkJ
nSsNTo+KNerR9vW-)^r4D_%TgshD%>+F#mCSU*tpk`sdk)1f}sPj@vbBq+ps^1D5?!HNfqR$zu5Fz%B|NEVm=o~koldcUq8b|<Zm
66e3d{@1a7P^~+|W~Zyn{%5eiWUoA6i9>uCDv3CyfbSdm=-%bGLFz{n0vLk{0tODN(c&TL3Qfr4k6v|X@qA){Ujch?+bvh0q>$9n
gF$}iHQ)qO5vde<;Dn*Y^jtr%LZ0F_?XCNro<WS~<v4^OyNW9V9umxE9&g`QOSzY6y<82XR24o#N6~jLe@>S-+8H2c3y`B4Xa*fZ
9Snj5!bw}Ks1RZ^*v7<Q+I;{*SoB(n<~VEFhiu3Wus6mVWIjx4Ji64O2Mg8mkzx1xC$L}WA>q%{;??>HSiGTx21~h5dMcST>XWn?
v{ZH<;43*EP4k_xTr%##*x3{>CIblVbz!raD<{9-vL#{v`txE|^ag;Za1dbYXtcUGM0z0%HVlH5h)Sv;0S8)E>h|?dcpO8=4YIg^
EL$BA&M;s`>AfpVOAd;mq;Qvy(IL3qe+v`Z58ojb5OFg*PT}NWmJHyqN-NL+>5i}<V22AOhPYpVCbl0K^#yiF$mJMCQ20VJpF*%_
a$r<!X<pM2Nz5Tv=eApbMG@eD7iSK~QGLWPoc<>71-i}wHK*r*CCMstazXv>RN-Q#P1GeToU^VNHEKAli>Q!|z-=4Uu9>8X&atCB
-nIwtiy|-5e1@bjWd_<VE*u<lj2wW!G`VRTs>*?i9jIsRHo=P|kNj%{KU}{`!?X^!wf}df_6x)qrw>=b^FVt7gjDNx37P;ev>7!D
hb_YtiDo@2KraQrXoi8#10e4c*a;nB+Ol`AJ}r7ztZF1OY{ZC{Bm<^}x2S_r`)R+dU6m}H{AbWpw#&{Urv*SK$f5Bx9$4M{Kv7y=
Hd<A5`ai~J&zyECasTO^?6grnPnBiOUN51a_P0S>$fRasbZg{nbv!bdCBWA055N{dUP|OM0rdouUe9nTbSo7b;H2BCgjuhkriuw@
G{E)U9(0@tYg%ZQ){)B}uw*pr9yS7uTSVA+9$zgAPJ5k-WL(X=mp|<<HVO<FhSJGd0Zk4xJ&9*+b?bwrA!VjzWhpx2_incXw+QkK
P#x+fy3(tRLV$TCN|NYfX}~d_>a&rRqko8+SKnT(^H-LbMAra4rwI%U`PTA4o9oOY;9HLz5ZvDtv=dTWZT|x-j=VsJTw?x`VRnUJ
AsrdMMToOIImGh>RQTbWM66K#+I@Z(MV`k1L@O!UtTv4pG;n~_Q0PpZEEzPkM{r-{!lmsI*rLe8M=sW1<f*|Jw?IP2I3SxDC_A@5
CF7fBB!>R?-5IQOW$EE}<@&%}5Bo?y95^Z@q<Mz1KIRJ)pUJC>B)7JU{|tbEQ`KrE8;LqKLvIixM7YdogB|OmO^f7bvn*)bE<Mx!
Dj1R80nItcU-ne<g8eIfkv6X}%rCZ>QUSfjfd-Dv4>053MD8kj*nq9X&cwt)=?KK$2KMQoxC9fV0UerUf<xc_2BrayRhEmEMMaQM
hFJ*)#OG?QVAanG9_Zn?{tRHbUUrvY7Ej(ugy89p_kgv7)l2&(zCJF{gs$q^dAw~BaA2YDUcX-{MEXdHVrn&l_@w9Q@*5NZP?9-0
s*Lrx0@wh#!}i68YxE8k8(1ERLKx>45E)orsZ5sHtOQPIX^v7&-k<)tlH@U=rDZz&Vn7iXeUvRr1SyU*)}Xj&?t+$}M7E{+-N$#4
<N;$>OrD{(^teBPPEIOHM3S+L;p>9lsgA=AM9FUdXR+h~m=d~?w1K^7jrax}-m8JmMn6p$^h*jB9QCE`>61m1XJf&jRyP`|487#A
e4G`4Aw9YFTw%`i3ZONt+Z)~%PhRyoHz*dn0?6DAQq)8G=V}HOZql>Jxd#4-W<(D>?d}Iy^fHO~9p0y8a}^M^Iamq|rmx`LO&^X9
XnEF&exdC?++xa;KpSa~)@w03Ib2wk<bj3)HcA^^D2C(+M+A8F+jDS>D6cYZMGr?|m{_hUEZ;aqY(KcWezJT6!7M4VSj@fi3Hawk
d5jVO*#kB@3Y5u5d^Ar&9Va=A(zV`9jzHx=G2-vb9e&WUk!ucPQ)6hfQ7?RXNoEH8V_vFG9xXijru_Ir@pUHx_d06VS)a-XmScXh
0H|yTkHELgdvcAG9HXqB3Aqy%=VLf>T0fJYljZ%BHvU(o;EyQsN+w)oD+Ai0Tw*_hktNEV2SC&IixC5B?MdXPs^4vf%5-EpP8TC>
4Ju#e_8FzbyGc;Z(9STs2pP!*43_X@d*g4s=E1Y!WXXsEXlG_9O5=f5m>NUR!wgj#ex&ZJG<x3d#+O-1m+0h(Wlt1(0!k!M5w*Zj
@-iB)6_?O1=6Q<Hw>J_lio8T8L!!5|>L8bX;zsKz8W_G7Ja`tMS;h<j$Sa}R?FB53JZ6ZYFk>PUj~<dda55AT!cUCgOgBvNNp4IZ
Ay~WDU$1#golA%ZZB*78YbyimjG^m1UwJOD{uN{h>DatGkXjshHbIZFso`t`P6UQ>08lm57l#w(fnRTS3<{WhJ8WpbSc5yc591$5
*6-Vl=EtbfJ%*-?;u|bKa%#YWhtt*F33^r_SnW3Q2RhZ|<(1M72Fe+^q&G2vz}0l>B68d%!f_#>YOUq^<werQWQ2h{pP+I!19!cn
5g8Oxmv9tOxy=u-{M`5Y`;Gce#<I--xA$b>>2sy!mHJd3QJlmBy^a0`m3I5&#+55sfhd8E0%tvlqG;&)XspNp&P;{{AQe1VnBan^
-41N^s|v`Bi58?^FD+UKB)<imb&L$?0F$b{WoisDH&F0)Yrh&6#Uoow%m6&RQB(pkNlI9eKaS+kek23{c0;Os*!}{h_X3y#J?jxl
oT}gfnlrHUP$b}x^ht82f6G9kGe(p82JHu5O0cwDl1~k^Mx@QaaEdEH+*CAWVJ5?wp@3)JiU%F^E{Zb(g~G5z2R8xCwHH_5v2zgc
M09Is2)LC4u5|~pV6C?lXYTstFe%RHTxTLX^`X{o@0i@6XK7M>^2lf~%@5GC-1qip4l5Q@f3TDTOPS-PADuVvs__fC0k&kI+GKPc
*-=UZo5goJe;I(m!UTY?u&2s*mB#_V0?1S|gC3wE2CcF;g0cN$cdLmq0{cyhoM~|5&jWv5s+zoKkf5IntSQ;~!_kHIY4h^yYT;w?
%p}8$n4tZzy^eG$HQKvJ`iJ^<h6<n%)9(%)>U90OGc(D=h}<K<5e`HLYQ$XOZq1_vY^}p~iWOJ)_x8PC*M9wSGCaanCfQ&nmf2oo
r~<e}vP}pb`^=s&cfsHtoi_EZO*g9DL=}YY1H~Tg^70B8+)@<`q{|<sxTf!syElL;`U8s(i=Rpo-T_vIf_L&$_3Itm;HS#)qyz)i
Fm3no$*=T3s{o&KJa)ril%+0|_4Qseefz~G0Bd9PH5%*8=zvVXc+)Kad3o;J2VWP*U?W}TKnqI%?tvviLaLCylFtnj`Po8c*oG3`
gaf}l_tX59<u*8ww}I+LexyMlqx221SQLgI8iXE02jo_Dz;Wz;fW<P%aAGmq0mqyH?jPNznPffn{)u-`6*n{pOz;?XE5JoD2=*|2
Q8-RH4iYk`G(j!NNy|+$_*E1MFARt=-lYu+P)i}`Ibed4zZ$x66&SNYg2^iDa<LTKRDg~C3<JKA0d;qA+%{PNV>t&_6LiJP%PXWj
DnzQ<GceCvx9@=)TR3ubfKLw)IM?;<ekwq8B}v#~lOCo3<8gDxH1N|UR~9z-cyji=r3ctn7q6_DwuBlEjqCx2kaTvjM<XyF*}uYU
y+XAbqdB;n1$ZU@KsKV=wL#7&r^zV%fRX?vIrr7YQfX)m)^7*R!tN?^TU|tk7${dY0=DK%fk=J8*N8<-ljWBDI51KamnGTTHTZG?
i#dNpG$4u?p9uht+Iy{(;m=96g;Fd4a`8`LcL3k!AS-P(*=!Dzh9_kVh!T{i6c^RWQhri?M`IZ2b^HGJnY_<nL%0YfT0O!1s<eni
1-0R47l=HCZ1m5ajOKO-hQ5Yqn9M-aXo=J>Gjn~g6UmJvV=~MSw1y5I@L-fdyW_V${TJxR0;p?bhLbmn;Yc721aF@l+m!6XoEOzH
cMDIyawK=F5k{Ox`XhLNDAnoG*{vJr{sSWzNXdcLuziT2uiBfZsau=PjG+XBR0<R(;7r@N_H@4}Uv(&G@x=0W@lV$oGIB_nG&eyU
5O?xmq=+iC1!?jlS0cICjdZ2$h(4_>OE3&-7u;p|0aU9j{3ON<OgI~Q$Mn9sbv7ol_;-o?2j2{(A1;YXQkz-jmzVVY4ofXXFh!Cm
<@2lhDIKsFf}yl5oX)hqzvxz40#=fGxcOh^^jz;~+$tX{G;OtIc-#%_&%a>+%64Kx4`|>S_GMlfG{;LMfIxI>>=2io`$bcNH)6yN
`qK9Ew}y}y(yR<l(PiFyC7m_qc3?137!epxh1PNRn%CfOu+dW5wP}FMa3kvc%{l`kC5(Uta^2g3>3Umb9-2@L4lvQ5-2!x)tqf!?
4{2ja24W4Z;O_cDpZ*Cv$;~+a=)eQpN1m2sWU4kuwkl1~&Pv`JRGVRr2I~i|CK;yDB60y=g&dK3)o8Ma-Z+ZOU9f*2l3@xm1O~~4
Ossa|<)r}JCs=}k0F(UUhm;QkLJCyA4C3QTYnGTN#*gcxra;WFZka|zSy3qRS3&!=l-iSD32&TIvt6{ZbL+NVYvTbNgJ`${K{BS%
8<6O&P)QH(`NCg?LQVe-h59c9h1xIaUMN%vlOKAdc!D)~ayp@-RnXHzrr0u?vt8Xgx=44ed7+^iT|@QO@e)}*^+B76!zCH4lGMAq
+<=6S`o7uCyH+39>OqeV0Ko%^%7g(Rav4520D8ilFaT;Cmzy}qZ(jUFLroG7=Jj$_e4K5%SkD2^E_nU9&&)WCMjnvt&=|$sy!r`-
${crtA|o~*We=LPKCHGBC*X1(;5^yYfQ!WhW81y<vcx8uEA14k7~m0srd%nx(F%eU42+txq8y+X8ISHz5^P?6T{yJVB481Y!9W=T
Iq{GI_67lUPGzOc;36@ZPVRe7wcfS*35Hru&XYy~h4ybRKLY3uYK8^Ram;ixDIJ;9b=F?BdG!+wl`K^=dO>d-SeF3Hk|%Ww1-YA`
1|%qoHa`Z^zwcR4zRyHWJXA*d;ErwdeHXx7jgJoCwV1-fay3@MesVbZ-3oKTp++xnRNCICV(HJ!w7yiP%W%y{StS9C8i5^hq>u7$
(OGz?o<V^EAqP0)hGx@{dw3<}DlbIEUTO9uH{<~}+r9jWhe{?(ii9COg?fgED)URxwpiYIWO8FPJi>;TO;x(?UjGdqDkvV!GNLJ-
50E2$@#dTLiRVgxQIx_jHh2?vuYcm9qR!z94p0_5vs%f>wyhIDUc<CnRA&I3?xf|mFMq<J0xFJoU{J<73h;Bvdw@bz&VjA;Y~+eE
41ml5$GqQXSa7Hw1Owo^ZGPCSZZ_5wb3NlEvW+kSWuQo;Y1?)`f5D*|qbpr!+3-jTAmlJ_X^9E}jNRtco+az87!H_$%I#kNghO=@
MOFGOA_%~l^}|Q&8f_$ygx9ftfH3gesL#0V+JC{JMvA+%DFy{wmVSufgTagEEVObH!af!T)SLH47r%S`a~8dJ#ZC@K-Vh+Y;bBw7
g9k=uz)w72DjUEqLUJD5ZuP(5P~BO?(7CKe4aE8ZeN0TVLd7&XNh3sMR9HPoos|4GKfuI8O@O|m=SRe7c#!hkWUs7&EvAX51%&{8
R3I8`SD;~`!4x9K0J$_BMmC1liF^(I)JEzDngo%79O&eAuv6SFL3v6LflIRPH5*z%1vMzKN%crp`shU>vP>Q{Gu}nrtwI+dYD3G&
eSkizng>CB1aL?eT!8f(Cm>}eM;WJkXGG)s9-vM<vrO+XY?ada((ad(WrK~uQenW^hyp)IF^DSKJj?E=cY&dTm`-<kAa9RPnh&^S
E%<_j{ye*hx}}G+LMw}5`_A{JJPYgw^4<*XC7I};<|#-r6&NO$02q(~;#LC(*qkPuw%3u#B$d|X<^aS1Zx!R*GF8zSV+N83%@06M
Hyi8StvVMFYPIxnljsLf`x-zV1o6R8MM?b(91m0xGq46Jy?yy>1gU~I8%m^2L-tjFa?~6%^`X$<tvzeVFeU=RoD8IM_v#l2Y5*L|
{CR-XE$t9!-0bi>0k1MVXIg+5)>q)A$Z@yXS}>>}j)N-*@|tXGFd%*hUV%<D<F6y|O@;|$1C;Rk?KWEr2sLW;AVq+Tpn&Cs!06dh
5hX_87bXNoh4w62&;RzgJuVa&aA;`^gWAcnuMupdL<404SZ;DxNjHIPEWe*_mF=f@;h?r?{IUc9fY$(GYybzHnFuPx=lUxMqKqOi
DHgWdu6S)skn~5oWN^rE#~M5=K7kno@z5lJk{*g+@nm4OeBJ&43kMaOQ~|hl@ZlNxS;UW!L1Qwt4j{rv@j{yFGzls1)<6pg)fb77
k+or}nodTD2CwVE9kG_}Gx<|0IJ$tAB!;$kzk!8>>dEgw519Z#V2+H`GkTJ~eQotHzzLu#MUMp_u(mz+O=0C8fJjFqtV8e858yq#
)(@}pLXu_%frj+%T8eSE>%3r41N7>a<<AhXSlr@`2)a`Hbu;eO@hCQR<}<B*yA8mCLCpc|bOk2z_8Fg$_epXA=6UKcb)XOiGPl5j
d4d<WUHI1o1EVa~@~oo^&@VMFYv;<=O*x5_D9Xs4cCyqix9zfj1!0q@5Dlhtf|U`pd}8?k(44M?c93v@2=S3J<nGMv!a-FC-pm0C
Amri~duFdIko+;8C8E>+k=`e@ShHLXzdiH6a8QX!2)&?uYKpp~R}bJpVCY2Uv=asxF8G+K3v+tQe(l+>IEwTD8fdh{Lyk3dM+X8M
_%SpQjMhv&V6T~eh#pSHpRKJL%l<leV+^S$)3z<&?vSDZj}0Ovx=ri_Fq5QzPCR7uyAz*zSzko}xzbyrz_#Un3HgXRt&RFzVbpE_
pacf!84dHmD(#m41%&E=YHMb$<S+DO*%P2zDE-P|>nrsfp<<+qTu-2AW_z#Cf<mPrO|}I58+kZ70qICUJ!+qI3^HkQBq&2`;sF#*
24c1Q0j3F-|K=?^j}jV(YD@<(Td7qdvp%`F5?N?x-8BR}`@5~C5k}o>rIjp0NX}e1g;{W~Fb3x56r@~fsVOjP3%zB#jo5-h&2k+o
C)CE)Kxkqx4`4&%W1KJ&dO2`@>Jz}dz4sA*wcDDo>1=xha|`wojI7-Ld=`O2%1j?q9Ub{{)FvF0&|-$>?!z&tV0*&NCw<@<K6rVh
6(|fGeg2(n89<R;+Axv`)Vi$iZo(K;cm47?MN$q(wPG=-LIbt}LE5c5Ci|FTVSS}|dfr`ljsQ8f?v<9^uiwA?`1Z>`zWnqDlJKni
+8|})MS-i%6?p<^j0;$Jz}N8m#k$Ey5+6#BE{9%XOc%g*OWGDEUIVm1iW**BJcN3n8DqnJ2Wfdynt3LR$fZ%9*nSNARS!}Dl46Tc
jT<1n0ax45Lu&(%xaS5qA$_xCly|G~g(eFC6fL|oNHLa4J7I&@o}ye~E@r3^Z@h&S&<WZ+cdy?sEH2oC8WlK53!r331h-u}uQGQz
LJ{Iz1&h>1{{^If0ik-}E1B+sl$P9?L2+UA6*TcYbFvD|8bBK8*`RfH`wNIyC`(5d4`3rSIX3Vlq)+A9fLiHiE<p|_B_M`EC?}@v
oy!XkOD--=M$v?ZK~WHa;66%4{-e~8Uw}YpDm|7SrM-6-!0TX;jM(H8L`uINNER@bWaQq#7>|IPjg&jb0FyD5(3IUGj^<r6nSr|*
!yX<(GW+}DQi41gcr?9QTIRGNA8Mz8t^^y~l;c{~NG8JFy@BlHg$Nn1w=^@Em<?B0>jBI&l#Gu$4mZE@03zf)CY0^uJs#j}(H-5G
bjV)cbw`R(#-0&lA5Sd;A_{rPf|r=;ZWFZ7WRq+UtEWR$N6*WXg?OIV?U3m@Jy|#*8n9P@<Zjm>^_t9sH-rve=rVC+BWW=3x2dz$
J~`oVtKdoWjLhv$W-KsSki<OW=u&@`=9&a4pM?>Mf<l;pOqNE<Rxylc!@}P6-*_-IkQTyBnMp&lB#Vmk&422IBhk@-DlTVeLfei=
0x%%TkkT?9pcF79I@`*R$3(3XDpuo{odcZmE2$Ui8i;{+0RrWNwmin3vZMBJd3mKD1uKw2x{ZvBOZfpp%>Ytn+%c_r^vZ}s>ZH-T
a}y6JUvJn4AR4S{LBTTdg?;Slk32BFdEfBM>Z8nbyip3EXD=|14$=}vBX*L~rvMnXQXt?D<cM~wzlA4TAQ|;wv1aXjtrP>GVyERT
oxrPSD~f|56KqZS>Dz0O3r!Z@QE9qL2OJCus6Vhcvq8#g8N>acQyNM}EwQ?|Jr`ZCwIs|Y-NMicN1GXhWt+m7t3ZfMI5)W=ks@&*
F};2CYbCNukv7o*X!QQKV$=5KGQqJC>lj@lt4m`CYoljwyZkIxTWJRn?qw6{GDj8CJl~ACFw;sOCOEFFC!V_E|8(Eb=7$m>!U*QM
!-J@6uUzuLo9TBzx0P=wO>+5=@Z6-x<X9osI;}IWa#HRlx!Tc9c*t<QWggnFpfWp6#HWJ%=#O^uD-XJYubY!{2}WsSsP8(0QgzV1
*k}k@%@hq1zD#&wEl1Xdhmy4k_o2%HX5wKUu0UldZhCnIhXFgAd?6Der~P!LVQI(~@5xnO=yK`$?XuA2viuFUrVT8Br?)gL09g|t
11#ZbWF_tmU1>{9bDF3E>eSPTmLV>U>@OJD+E~kKpL^Lzq&1qq3J?t2wOPI96PSL0CY}Bp2tr{~$^E1Oa1%46l56wMis;b9>|Xz>
4AQo46jM+l8a6p#C#@Qe6AZHx-HCIV33!+e+U-o&>5Ni2AiY}YswSTn0jHMALTtgT8ps@8>Z8N0oDD2#y8vDBSxpgV$woJX{)_2B
oe2fS^nd%A&na?YVK*HE6dSy0Np}sF0coJ1c&1UvAs8G=V5~;^FB>q}@}>iao0_x!x{D9yLy|ueiYZJzue@Vf*q4{QTwO-;r9WuG
V#pB_0(TDmxq+qvVBZHG3a&&du=Msc@leY3o^9wrdCyE%!X_05#PIFUys-|{ep=nSQrZT&J9JBpP#jiVYbalVY=$4TVXvs2QF5&h
;IGXR92Up0r)5|V;brTb!6+nIW<Zl=3)=ul_C}0G5!v83`tIdVe>5|rPnH_|0e+Q<hD^r57SYl~C<pB)!xJb;2>Wxw3!aq*ILdXh
-45(V0OC5yNBphsuJjCYtDxad<gMOYd|p=|Xf0<A-7!{h*|b6#eRH6agORRFinOXW^@q-)qIL-ky*GEN52!%po!NO&*zxj;Gd?tE
L^I=DRW21^N^;$eDiTZXuet;DK!9Wvr~9QlZtggA2h(2wB1(ZcL+35W&5x7^(BbaV9ZWAe!>3mejzuUGolCz&JqR5=f&A&{Y-VcQ
XP9XWyR7U^rY~q)Ko?1kCdkmNwW=HqY>j%V^|I<o)O$%#Qm1@Zg4FvSpsua}M$t0dDBvJa%_s#Vb^1*TjM5_+2K^!&Ffkbf*Y5Q%
|1TwDk^rDo2hs$&SQXgOqL=epKY$bvBL=xAZj1Y&rC2Xg0b9}|$0vA9JV;x^y<CtYH^=}TL@#7-a0e+Q+@n6gR<z*@ns9qgC)(oE
EA(E%oM?q9px-{kcmOKlDM2e7(`g&pBT}0_zg%W6PuB`6p}B#v=AYfsJdaH80IjJYC%d=S9vwUqs=4JHk)#&@B;5eFDc}qSq6>uJ
*;q=30C3UR8v{eTp<os?uzXG0hva)`PaCv&jmnZVp{_J})T1pBMgW@A3i!IagO>#joFQb*SR00DBIq!TvFZ_6GM&^j6&9(jH?W>0
%G<Ypr6Lpk>|Ktv;;rVvE(-?dXQ-}p1<c^Z=GB7-7YrS^`vIo!$2<_;+?gk8(Lw<E|5^gQc5%jz;Ko(3o#bQH(l#i+PQ#z&OUPb_
6R-o<TEERqE0c)UOgEEvH4`bN0W)-89U~liPNV261#F@h7eHSkjR8~<_Kv($vAmsDqX1k+B&@s>WT!N%YlrL1JB6G3ndZw!R~ROT
v`NJngzm>DKN8+J=A!wWeWcMGEk~ar(<D-S^hEg*Gf=O&4qkQuQ()h0D8aV9AGEKLJeq>!&&xs^CIo#rim>S5hCn1vzeC=&_k7rh
Hn$IB_vQG}=Un5-Ax4JL&`6;Pt>;7a*+oGQ3w=aVGwr>R2wp#W`q5~la)7zX-YZ4`7l$SaX22x;Y;9?iP%|`0HK`>@x8KL889S(K
U`a#54G_}HD{_4}&=rT@OJ7ma=8MGI4TMFOcXd2SCov!;Cuunz{pOY<Y_X(!fsL~U1GR>>`uZz>*75jbh-DxnD?V{&Sdk_K_;3u*
R*qaAd0$#qQ8d(On^ew@u$5@kqNoD7S~l)AdOvw3px#Fh{#N5jhW7O|1I5~t%~Rxl9}e(AkVDT17AZJliRV|&DakAk9hU22b%^tu
)@%abSGPXd5xd@g-`q;OMjk_oE?dux1C#FNj@qUu9Sn|m1<bH#ywCWSI=G$QC-csp>MAWfprB^MIe=OVAVrdKqZ!OG?FwxIeQl}b
yD|abI~2%}3q|QzPNc)<m#o(n-cSQ{hP^EG^ScyP9mCk9;Z|c(zPZ79tn3<H$*kEAfByECcC6%BTP20&Lpy2l&sL-Q84o#-ie$N9
Y;P5$`(Sir`l$|;UK!<h==C6RlpTi!?!G4J&<=*2sAovO(GLFdiVc21&H-u_X(y(M?3)c}u>!xjWNsFu!qLdYLp$8&7TUFO)9yw=
uNTq+3DV*|;P!8RrM<OKj*#fY*Ht`(Q=$k>3t{@5WO0P_vW+KZ*shW7CbUa;CZ`ljCYlLi&F02g@5Hk!bewzUJajq3OeLMi7h&qv
nGBsgWVyzdi9tZXpJG_?<;^W8cT|VllJTkDhnm0nmG;(Z-k`omzb6ReHndCEz5yHqKw1m|-1V|D4AQG+og%~w?>W8rD(ZLsZ`AL9
A=Gd0FXaK)Z-mNI<@e{*Rvpve^o<y_TLd!dJsF8@!LO?60DxOnlt$yctSYB68=9miHtMjZ$y023(Xlh0L710i)$g-(0IAiQm_1@(
0A#r5SM+UQ`#e1abe&c3A<l2cDjSi>@DV;vLq{iN(Fe&)%nn?x&6}QsDl(I>51_I_H`hU4=JjVx*W9UC?tAAGHeHrM=8TO4m7Bb=
qiul9Yi2!TzbY^dGSymR@dkPJt`cxJZn`8=kq%rUuq6cuQXK71+B7CZM)($h<Z02oIyg@A?&YV2-!0n+0c0V{2YZ~BqCg!@%d8`D
#nio2e24bZa6NV}e?q4#rbslCcn1kxV;Yp8xq`eIWTi$HC0f9R3Zcm|r`^k+=;;(Id-TTTTPbrh&036hy#NH6-9SQ>AQ`GbX09)6
-~I`o?&zLZluxzNRRcn8Fp1d&PIqG(-4SHdI>+=M&e!hcw}IeIgB2xvI%1+TG#fBc^*(T`wZe(vT?)A*(iXAR?b0&=)O8pgXwi(^
LTIh!!{rGF3k0qc#x+w`ltU`b8JL;gz5D^7o(>ikQ7sk=8U|#6&`XEc8T!Hi1A|&uaavBmdc^mfU%2xDCWty?7NBHxU<b)86`g2F
5(EdX`7~jGJuM-H{%ITQ+P(Y<qAva73j?wVzSmQELLLr|RsxoPU?H(u90B<_`CS{X?p%MIFw(sdesnl&!LtX<35K;|1R3C_oG8-G
n!gjpfspqve}bsf=6#YfKm%pG1_w~@1c3Aw_R#<yi?M1c>?fxROY`n0$OB0F1V638$x*EI;TRel{n98TT9dD6Y_Za8Qy{wCzWy@l
U}+BnbQuP?<jsERxB~iKjjmw<d6Uk}5IgwdeA|D7>&+81a(+OJce**#VH5~tyvVsvB8LD61(=+2j9S^RLMN6wePqSRnp>BaF&Qf6
0$EhR#N}Xe2<Rj(K?Jq`4NN#!DNC}(J|$>7y<w44YhnaH8Pvys6k--A!>S76v=6%<V1TIyd^j2_^cPGYM&*!jOZhJ#goD*yA{AuQ
vdy=HtF`?BCYm~2$NXLj-HPr;@X5@k(FQ=MI#|tVJST*%Bv-df&_YvZ3XkSm8eD3oDD4i#l#XbauINeo7`^XswY1K-;k(jT-T4F)
PTdCw%uWOu(qed7vTB9)b7(2YN$8MsAhzj+Ie_h9%)nCzy4zCG3gMYT9;iZaodf!w2yWtt!_J1bXl(?|?|*_>MpjT;3?KtUlMuTO
0qEThL}_p;#w@K3dv-<po>9IS_N$@EE-UR0QjipU0sS=V0e~R{7mr+<8+IUM8evPKg4bJ5`_<5_gW9qao#~Ac9V2yMG??Lw3PV?)
dera{GUHA1Nw+7m7OZ-A3@7`XwOlDkJH_yRFb4|2Yo3-M%a!A!AutEC-IoY+grw1rE+;2ZAvp&PVR~>D%rIs)y2!{Qrqbz2@D^q6
P9@HHT={5Sa?)r7s<ifbFj-o_8R|w*JF(522fY>?uDai!UfAj##%hClN(#RE(G&%wvlrXVt_oN(iDYzit=VY16<g5i4bH;Ak4vm+
k}8bSxoUO)NH5v!U=9xLS)EPJ`|S@9uMa?v69cP>;F3;^BtH~FHwy6+njd^C4@f7xhjqIKjaNVx`TYs13^JhTC6Iw-z`SQ%5R}~s
4IJbu@i7{6_CG*gU5Ik-Dee-xqb_|tI`g<p<0^1GAq*Xj9IgQt1?AiRO2%yEf_^0%w1NSPSVE#g)@|>k)2Th(xJsN4g#xF0{|mU<
V45YF!K2qex&@TBz(}{`Tv7gLya}nGd7#c|c`%UeeSm3$3Be@tw3lG7!l(#%pdoc!ixg>?6#>9Rj3dEP*xMgrTH(s55;*rZsZxMw
_O#$OYZaj_exNc$@-DEO7Y1v#7sM7Y8Z2Uta%6<o!~kj|-Ve>AE2rfPoyGw9ci`T(OVGks50Mli{TYn@f=@y?2@VN^r&WKHCO4Ef
=)(kl;r5(jS_CH^yvlwOPssRE5AO6C0fNS0+-RlO(WOr^7f8Iy{+iN)Sf}|3NEIleShjJ5Na`D)kdbWq<Oq@EgW?~Fi0gJYVqvUz
i9MsfjJ?|%Xb|r8Op#U5OHbnf16^WI=$((-ZNS1<&(JPX)u<H+>3XS0rfi|(+XDk80~AXu>Rm>e)+L?l{U`X!EmnxvJ-r>xxHI%l
IH__aI=P3D(<NAw_G!j9TuRt~hNl^}hEK!QqZyP&JNZG$koD+#Gu-AAroZZxCqjnM`~Fa1!K^3QB%Cu+<?Jxo6c!FXOu2G&8KNr3
B#}^YdAlpOaMlYL7J)uN2Eue}9Rfv+onotmJ`OMieWEiC?Z_#X{Y^DjnDrD272sfwrX1{g808fhj5Lprp#sQnLFbXm1qpAv6R~jC
qr|y@g=G9~`X`zt3_GDHcxh6lLy1DN7n?#%>7!Wo_to44!#MbjAs-{kp;?8=s$5CIg_(|X5lkwNs{w@B_9MJ-){!82_K^~0=rAuT
oKF=-lVELSapGjtwO$~qwpsS=hHL??*NSXFQH?#W<Rhro18EPhKpyP{q;qLD10oSp_w9yk0j<N(7&-{f&I02iPeRrzGzWN2rZaow
bghGu@V|k0x80B}sC5OtnT~vEPS@zjqGOH%9>}E8Kq|*E=uDvLEc9I7cQEzNC#W}c1A_}nddM+-c<5+jn`uY^9(=?zB&id0KTTog
{;b5p3730DFA9K&&KY(*Pat=?L6Fennvg5mQWPEs9Da-KPtdQV6#)d5h@%$e^3e5)+AmbH8=mQ~W>ju;I_N==v$wxZ(XEgmfTc;^
AD6<gjhL-yjF$H>#&4X5)R*yj`9z?Ur5$bBjL4peaw$fwFVMVU4bTAONq`#Y?1A~6I!9~?D;Z}7;lJKDPverDtQzR4fa<7TUb&KB
W&jh4UOZp{6y_d`7y-yY07%!*nnwcij_@RxlTP46pc!S}F%6((4{ngFjOm^qqsFm*rTuTQ4!kZ6&oPs!Xh|(FqDHrBpg)rQeeMSx
916whIVpp_q<Sk4;Tw=a+%C$CyPR}*f|-Qi(DG)7L0bdGq-T<#j3g(d$C;r;ZzG`FpI}wnQ}iCf3Xjw1u&MzUfbj!fO&Nt28>6uQ
W3q9-W3zzQrP-itgEz8h{di`{@l+|(LLX;_0o<5g8rc0z$!@Q6m6hjrRsl-2Sy|!&j3o`WIWY&D>Ks{+rj)Bsy9pg++m}B}?KBC?
Pl3iQGeHP-0$>U7|IukX>ReJNfw@4kh@H`f-Rr+XHKt_4dPIXIG5jp=QEp~~vWLd2F^GoS8DXwql4!pQnjVesjK`6Cs(3It80Zuo
WVDpP0x`KCs96BKtVff-R`;KP#iL5R)WOY?yVwl(i&Tibcr+*Fi!rvP_@%W1U6DNIw!<F)?b>2J$phd)5TGSPGUw+R+v#ADLS!+)
4r%-WVhfPL(C(v7C=$Y4VfA2KprCGFxdb*@A}twQvBZGBnc6{r2p#Y4>~EhgK}qZTbxv*pW5_HHBjaOoE8pC60Osmhg551u_uM6T
cyjAG;;S~)#<^|i1DrRufPIRU<R$1+0NIr@M*wY%)f$K|q)4{wk?NNrrJpfXbOa$biB`oFpr<-I-Q5rf26o#y&JNVY_VrI!rh8-;
+z`4!EUc(Zc_^-ijMk6k2eN$D1ArUswkweZAYKixikK|`hi?-(z*tF>7?;^Vj?f!17B5aihp%P(158IIaPoSmgB;=nxctnGH~FXO
C{5TZa(q#-9ogbu_aA|UAzo1wPz+XNp7GlT_nEXG?e2`EoxoN4E6Lvn$*k-z*4F9DSYix*hkCMZ8Iz2HIRc_!JukzMHXx9}3ZkE6
f3<MHPQzzG#lKPxY?Sj2^Rvs7B%*U|G|n1vo)rtLH7{?^@h#ZtXf!#n5}oXL7*)6;V23iNey~9Ty#S>&brpK<>K0c3-DfN!RHXvA
3IPC6kO$!bfW}oDEn`sQLf>-6H^O*`3wI51XNY+kV>}(`>YOewugKDn`;p>H_^~qmh5*DkVgLfKBCiYlg?R{l9U!(75&uJGNP69I
0C0Jmjfb1I)B$?zuD_BW0M5V9572IcBXW>{Hjb}KTm<-FJLqVyrb<E?ZBKyq8A-Y)uGl^U>XsF<X1GKqJpiy)59TwngPlkY&Y0(i
F&fa0rm#*1cDoU3(~CjPwW?#qv~Z|2tkeW^6%!`~2G(JN5*dw9sJ_B>u>ApMsVUXr=Qg^Aw7Gf$Ywz=5fJ&s>P=XmOWZJ3qAK&&n
k_&m-D|QBAN!ktfr4f;KhC)dXP-@B(d>>>znsPu!uKVZ6uF%uPJ0n!qku~H3Ev1p4#&=8%7*8Ay%h?3$XDKP~(Bf>HBa>HYx0kYX
_%PuN=P1Q!=fT#he#LWvi6-Nrqi5nS{*zAqt%BfB>$05AmsjW$@PPn6uo*IU(Y^>vULr}`03cfQC;j^MKIu<-CIP^Zm0LJ&H@7qx
#c)u!iPkpJ-2*U+cWH6$CZk|SaE33!AaI58m{c_bYzcz`KGfN_hi_{byiDu$?zUH~7Xo!BLbH@;Ks_|DIfC)aWq7*WY8hY{)j;={
*w}A}763KagV{0P0)weMf{c6toQ03#W>yPTnRrNtptP{%2m+*s;r9|=z_n<pMOv*spiqa>yQe#;8x9>RUC{#^=Q39MTPt5aaER{#
)6?V9m7MLqjoaZRzjhV=DS%51X%Z4XTM@Sih678<mX3Y<@N%xMH{uC0$a+#Lumi{)8y|jpXKwp39w$4Ar-!?p&;EwT(xCOCb}bXR
K!gFK`nkv9kg3l$MyKsmRF-X#a$K1Fg{T^mIiCJ5U3X$^Cuj4`=V2Lf$IGkjfW8R}k$_HYJb1g_A}7kSOa|d1E#e}Fg=Cuw@cHSZ
UAx|xYYa0JA3*w%OSCP1`eXJ>4{Y%tV~@8Z)29zxaI}Vyv8yMJn(<n|o)v6Y%K>}<8BE-cSkpt>S^)Uf|1~@oKj3s;UX`_FAyV`n
=9O~;+RZ(pfjH#~)9bP*+GmBcI%v)$AQ@5l&7Eyu6NIJ!-x;U+5Z#+!E06B$eVF6`@sN75Yyy-2=P$q!tb=4I2Q*f6+hfqDST%zN
3RjXLP__FvIRG#W4v@}g9j#R_uQJo=XnLl2BhRMVismiw%m`o#BX4Q#@y$)@++%=v=nNqpXnXwh&fNARaXiot1Y@P!Fa25B?e5}i
SQRoM1=xwSjVY~yrhqt=$W?y7T+6YqjC!^ZaWnJ~<)LjgTk4{ve3<OIyna<HVw%F22!VohX}Q@~W~CqigbOY)EOqXyJpAijhU{`J
20ZC7W%tRg&wcF&?h<$zdRytO)h>Ya>~YdkQVjs>FyIo$9(&+1TZB<O6L+)6{h+wd2sB-NF+QkvX_P>3HYSV1Bs_S2H30fXny7(Y
4i@cc+UBRT$r-BGhbwUR&0X5myUjrY=nQ>z=-t~}n||Op(vAiCLhX<4@$0hdO?Khvfn!MKfDZn#t)Y?et<h>0d2XxTfrN)1?Iiy<
KenwM$jl~RJ5T0|=ht}2&}r(uMMC19Znu<~hyZ$}PH=W*Z7J|=)`BgkXKv}uea*aqg-*kaXH@^(zmHFU?ajs3pJFQ=1yDE@fmTl#
cJ`McrygwVOs_ahl?ysFoQl9|<cCzlN&Ia*gLvF+!s|c3Dnlz3FGs9(OE2F@Ek-L+e1VjU3iB1O@a8TCL}=IG>ZFsTCBuuoy|Z`2
ssVEBY;0jX`k$8CU{c%8WzxPyk!|cMm{laXWJ;S<MVcDdnB6i#Dcw;IX<V=eBBkKk0zgKW*ROg=mgGRmACL*h3%n6seOh`;ZlI%i
eWCSsJq0OGrY8-ySfb0d>zz$>>k*P~GIo~9WAPuK{91T((WUHWN(d|`kcPFJ=l#Rc;Aw$qu>c-??r=K*;UNhDS)5DtA;r+m<K$S&
^NEw4>e<CW6$j2*oqt9aZk^n2gL%?Nh-qF1t%IMb@J8$OCJMH_xAz?eEts<*HPX_lY@hzvn>e|&FyWIpBhdl0i1E#7!$|RPOpxf@
^E6_K8DSqzxX%YSwSz<lH?DMG>JtF<`BlBoz*E-2rdw|vFp@M#V~42$<h{AC9j-PC2i32Ay}k3;*AB>IMXlJ!%Eza__NY4eFU2;V
8W+rv0{EnaRw9{(;KEY`8=W@LVMT`ICT7VEqQlr#9@=EQ*kB{NE>n@dynZzwU@d0unyek>Mm6Zea8{9SvON59#%^yiov~tc&48b@
Fs$0`orhH`<SOv*W!&DzQAnrCt)S+~YnM}^!!n&*Gt4q1AbI9or$%T@{AvaIOMm{@zA~hN&cV@F@Z>MAUj@}ElfFYeMoKSFq6ez(
B;6gE>uZE|dzTeG6xu&Idd4r8p5FQt-V6|BIdg6Hskha+>D0M0P?4cNc+BKx=V{m)oTZsQUY*7^?Hc(LkV;v%Y~DkL4F)GbIi$B0
cITO0cz(4W;M--J({RuUJafqcpQMhIvcI}yh5P_m{y8m|?B>Qpcny?ZHoMCuHoNxZ$Ku-vZ@x@hR*l7-Spl=Q=7#C_VdHXA89Fi3
4Wosvbt8NRHS#fs0k>HZ6mxWLg8aO^8f7zkHRdxSvi!uL!R4tJO{1~=$E)Jy@Fio`DNFI*-gqcpYD!LiXNqTd`SIzGy>)h>=Tmm!
1m?H^a1`4N&>$+P5K+@}6#epad&IDn28}{_EDC1&zDJsYrQinw!i$SQ;YWI~6~Ho|cALK8vjBFLKsW2kJWBk)v8L@WhOxn1yWV%C
Z!i{;CWW15Fx$r`KlV0hv0hAYVWkCQ0PY)lK<2LM)I`&*u3s}YG7R04HK0|d^w_2vjG6t%@Z>J9UzL4T{0H8;{1<KCa6i{QbiM#v
0z+WMX!W?c%i7mmt&Lks>{P^0Z+%TS>Lfsfd?tt9R_T^ga2Z?C`_mf)n5(SxjVF1-TLvw8>ReDBBD2eib<#tzglDdnW>Og4>f~Q}
d6n)0`P|u{dnm$QCD%TzolUG(DmShW4U`<83R9PThG;i;?juBNHfdKNzqYK%$0tA5-bQlk<)7^;u<`VmGqBKBh(<?kCMhUq!I$vh
zDAS_n1fWu*1kTw%93n5BFa&&;7yFir={*~AQx2Hn|;0B<+`uXTYJWlPyU{#cOHINqiv2B&^q0pH<a8uB)5SlsyzZC)u*X*ONK5-
n$0+&ubT`TEcq+B>Q1)!p-pCby!&X3R<4Zt|MQDxJ3aVzkwNTolXrw4m%-0Oj+rH|Exx_U++qq3vJhCnQ*@u)+1w(c;0RDA=+q-0
_@_nptfJYrf>|$BmYqx2q!=hLAenWhYGzi<Q%7dt5EP{Hh^y=6h7#Q+jA5yMith8P!MMcGItA+6l<KYM#sq$8wi;zC*ZC#>Kpqam
{Bg<d_11&tN0Cv5T`<%1G~%?|Uwd<{JL+f>Oub7S*NzKz&EW(a67aPBwq8~ahx|TOD~OM#rn+;N5hb!;*{c1Q*RCmU%3ygW_R`UT
Wbg6j`;6wxj)6zEcKGHdY3?vP-oXWmZg5fE-r8>a5r|Ychx9f=Yj5M|Qbzm3UG1Q{D@5wlOjiH_DNxQJb}={hm^G6<He^DRo{;Cf
<g<$;jASnzJ<kwl->6(d9U7zMb5ZCii^Iy3c5XE0n*N9%?b`LeL*>fY;s0>zPABZiZ}m49+kT2|;29ZzqJlE^7i?AL6r+t0=G;f|
w?Wxecok8rf2d4J^A96@0g_R9@1xo0#Z{Cb&HSLOhc@1Az22kIAxA4NQNxnk+uKaJrQ0aURUUl<@!73MLKoU?53)ueknrwL%Pn+P
&cdORshf02lXI+NW64?v7}yD5>fBhl6~LTh@|ZGytN_e|!|$r&V6FL=*RN)vzYKs7%*1ZBnU&EE48;STPHwDTQ;s(`SrHmlF_{jy
Z|h9kT<^S6j-Vqno!Ep2;or2Q$SA{1c(ok{A(6-qt=t%W0yHdAtf;5=MT3RR8o(H~R%a=v2XZ)GCzx4TrmNFT>%O>}4uxMZ-q5(9
Krr7H**(54%@&~4<dwm>y~oVp&>^h5^l)bV)6*N5nKg%|T82-??Gdx~m;dko2P0zy*#
"""

TRAJECTORY_IDS = (1, 2, 3, 4, 5)
EXPECTED_TABLE = (
    (1, 8, 8, 0, 0),
    (2, 20, 19, 0, 1),
    (3, 7, 7, 0, 0),
    (4, 11, 10, 0, 1),
    (5, 18, 17, 0, 1),
)
CLASS_LABELS = {
    "strict_saddle_robust": "strict",
    "negative_curvature_unresolved": "unresolved",
    "non_strict_saddle": "non_strict",
}
STRICT_COLOR = "#2474B5"
NON_STRICT_COLOR = "#C43C39"
UNRESOLVED_COLOR = "#E69F00"
INK = "#18212B"
MUTED = "#5E6A75"
GRID = "#D8DEE5"
PANEL = "#F7F9FB"


def load_embedded_csvs() -> Dict[str, str]:
    """Decode the embedded source CSVs and verify their exact checksum."""

    encoded = "".join(_DATA_B85.split()).encode("ascii")
    raw = zlib.decompress(base64.b85decode(encoded))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _DATA_SHA256:
        raise RuntimeError(
            "Embedded trajectory data failed checksum verification: "
            f"expected {_DATA_SHA256}, got {actual}."
        )

    payload = json.loads(raw.decode("utf-8"))
    expected_names = {
        f"saddle_trajectory_{trajectory}.csv"
        for trajectory in TRAJECTORY_IDS
    }
    if set(payload) != expected_names:
        raise RuntimeError(
            "Embedded trajectory set is incomplete or contains unexpected files."
        )
    if not all(isinstance(value, str) for value in payload.values()):
        raise RuntimeError("Embedded trajectory payload is malformed.")
    return payload


def parse_trajectories(
    csv_payload: Mapping[str, str],
) -> Dict[int, List[Dict[str, str]]]:
    trajectories: Dict[int, List[Dict[str, str]]] = {}
    for trajectory in TRAJECTORY_IDS:
        name = f"saddle_trajectory_{trajectory}.csv"
        reader = csv.DictReader(io.StringIO(csv_payload[name]))
        rows = [dict(row) for row in reader]
        if not rows:
            raise RuntimeError(f"{name} contains no rows.")
        trajectories[trajectory] = rows
    return trajectories


def candidate_records(
    trajectories: Mapping[int, Sequence[Mapping[str, str]]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for trajectory in TRAJECTORY_IDS:
        candidate_index = 0
        for row in trajectories[trajectory]:
            if row["event_type"] != "perturbation_anchor":
                continue
            candidate_index += 1
            source_class = row["classification"]
            if source_class not in CLASS_LABELS:
                raise RuntimeError(
                    f"Unexpected candidate classification: {source_class!r}."
                )
            records.append(
                {
                    "trajectory": trajectory,
                    "candidate": candidate_index,
                    "outer_iteration": int(row["outer_step"]),
                    "gradient_rms": float(row["gradient_rms"]),
                    "lambda_min_estimate": float(row["min_eigenvalue"]),
                    "lanczos_residual": float(row["lanczos_residual"]),
                    "negative_curvature_upper_bound": float(
                        row["negative_curvature_upper_bound"]
                    ),
                    "gamma": float(row["gamma"]),
                    "classification": CLASS_LABELS[source_class],
                }
            )
    return records


def build_summary(
    records: Sequence[Mapping[str, object]],
) -> List[Dict[str, int]]:
    summary: List[Dict[str, int]] = []
    for trajectory in TRAJECTORY_IDS:
        selected = [
            record
            for record in records
            if int(record["trajectory"]) == trajectory
        ]
        row = {
            "trajectory": trajectory,
            "candidates": len(selected),
            "strict": sum(
                record["classification"] == "strict" for record in selected
            ),
            "non_strict": sum(
                record["classification"] == "non_strict" for record in selected
            ),
            "unresolved": sum(
                record["classification"] == "unresolved" for record in selected
            ),
        }
        summary.append(row)

    observed = tuple(
        (
            row["trajectory"],
            row["candidates"],
            row["strict"],
            row["non_strict"],
            row["unresolved"],
        )
        for row in summary
    )
    if observed != EXPECTED_TABLE:
        raise RuntimeError(
            "Computed results do not match the rebuttal table. "
            f"Observed: {observed!r}"
        )
    return summary


def total_row(summary: Sequence[Mapping[str, int]]) -> Dict[str, int]:
    return {
        "trajectory": 0,
        "candidates": sum(row["candidates"] for row in summary),
        "strict": sum(row["strict"] for row in summary),
        "non_strict": sum(row["non_strict"] for row in summary),
        "unresolved": sum(row["unresolved"] for row in summary),
    }


def write_summary_csv(
    path: Path,
    summary: Sequence[Mapping[str, int]],
) -> None:
    fieldnames = [
        "trajectory",
        "candidates",
        "strict",
        "non_strict",
        "unresolved",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
        total = total_row(summary)
        writer.writerow({**total, "trajectory": "Total"})


def write_candidates_csv(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = [
        "trajectory",
        "candidate",
        "outer_iteration",
        "gradient_rms",
        "lambda_min_estimate",
        "lanczos_residual",
        "negative_curvature_upper_bound",
        "gamma",
        "classification",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_markdown(
    path: Path,
    summary: Sequence[Mapping[str, int]],
) -> None:
    total = total_row(summary)
    conclusively_classified = total["strict"] + total["non_strict"]
    result = (
        f"Among {total['candidates']} stationary-point candidates encountered "
        f"by PROBE, {total['strict']} were numerically identified as strict "
        f"saddles, none was identified as a non-strict saddle, and "
        f"{total['unresolved']} remained unresolved. Thus, all "
        f"{conclusively_classified} candidates that could be conclusively "
        "classified exhibited strict-saddle behavior."
    )

    table_lines = [
        "| Trajectory | Candidates | Strict | Non-strict | Unresolved |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        table_lines.append(
            f"| {row['trajectory']} | {row['candidates']} | "
            f"{row['strict']} | {row['non_strict']} | "
            f"{row['unresolved']} |"
        )
    table_lines.append(
        f"| **Total** | **{total['candidates']}** | "
        f"**{total['strict']}** | **{total['non_strict']}** | "
        f"**{total['unresolved']}** |"
    )

    content = "\n".join(
        [
            "# Strict-saddle results along five PROBE trajectories",
            "",
            "## Results",
            "",
            result,
            "",
            "## Table 1",
            "",
            *table_lines,
            "",
            "Unresolved refers to points that could not be classified "
            "conclusively because the Lanczos residual prevented a robust "
            "negative-curvature certificate.",
            "",
            "## Numerical criteria",
            "",
            "Approximate stationarity uses the dimension-normalized gradient "
            "threshold rho = 1e-3. Two independently initialized Lanczos runs "
            "use at most 30 steps each. A candidate is certified as a strict "
            "saddle when lambda_hat_min + residual < -gamma, with gamma = 1e-6.",
            "",
            "## Experimental setup",
            "",
            "PROBE uses 100 outer iterations, at most 200 inner-loop steps, "
            "five conjugate-gradient steps, batch size 32, LoRA rank 8, and "
            "mu = 0.1.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _hex_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def _pdf_escape(value: object) -> str:
    text = str(value).encode("latin-1", "replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_width(value: object, size: float, bold: bool = False) -> float:
    factor = 0.54 if bold else 0.50
    return len(str(value)) * size * factor


class PDFCanvas:
    """Small dependency-free vector PDF canvas for this one-page report."""

    def __init__(self, width: float = 792.0, height: float = 612.0) -> None:
        self.width = width
        self.height = height
        self.commands: List[str] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str = "#FFFFFF",
        stroke: str = "",
        line_width: float = 0.6,
    ) -> None:
        commands = ["q"]
        if fill:
            commands.append(
                "{:.4f} {:.4f} {:.4f} rg".format(*_hex_rgb(fill))
            )
        if stroke:
            commands.append(
                "{:.4f} {:.4f} {:.4f} RG".format(*_hex_rgb(stroke))
            )
            commands.append(f"{line_width:.2f} w")
        paint = "B" if fill and stroke else ("f" if fill else "S")
        commands.append(
            f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re {paint}"
        )
        commands.append("Q")
        self.commands.append("\n".join(commands))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = GRID,
        line_width: float = 0.5,
        dash: Sequence[float] = (),
    ) -> None:
        commands = [
            "q",
            "{:.4f} {:.4f} {:.4f} RG".format(*_hex_rgb(color)),
            f"{line_width:.2f} w",
        ]
        if dash:
            commands.append(
                "[{}] 0 d".format(" ".join(f"{value:.1f}" for value in dash))
            )
        commands.extend(
            [
                f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S",
                "Q",
            ]
        )
        self.commands.append("\n".join(commands))

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        stroke: str = "#FFFFFF",
        line_width: float = 0.5,
    ) -> None:
        kappa = 0.5522847498
        control = radius * kappa
        path = (
            f"{x + radius:.2f} {y:.2f} m "
            f"{x + radius:.2f} {y + control:.2f} "
            f"{x + control:.2f} {y + radius:.2f} "
            f"{x:.2f} {y + radius:.2f} c "
            f"{x - control:.2f} {y + radius:.2f} "
            f"{x - radius:.2f} {y + control:.2f} "
            f"{x - radius:.2f} {y:.2f} c "
            f"{x - radius:.2f} {y - control:.2f} "
            f"{x - control:.2f} {y - radius:.2f} "
            f"{x:.2f} {y - radius:.2f} c "
            f"{x + control:.2f} {y - radius:.2f} "
            f"{x + radius:.2f} {y - control:.2f} "
            f"{x + radius:.2f} {y:.2f} c h"
        )
        self.commands.append(
            "\n".join(
                [
                    "q",
                    "{:.4f} {:.4f} {:.4f} rg".format(*_hex_rgb(fill)),
                    "{:.4f} {:.4f} {:.4f} RG".format(*_hex_rgb(stroke)),
                    f"{line_width:.2f} w",
                    f"{path} B",
                    "Q",
                ]
            )
        )

    def polygon(
        self,
        points: Sequence[Tuple[float, float]],
        fill: str,
        stroke: str,
        line_width: float = 0.7,
    ) -> None:
        if len(points) < 3:
            raise ValueError("A polygon needs at least three points.")
        path = [f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        path.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        path.append("h")
        self.commands.append(
            "\n".join(
                [
                    "q",
                    "{:.4f} {:.4f} {:.4f} rg".format(*_hex_rgb(fill)),
                    "{:.4f} {:.4f} {:.4f} RG".format(*_hex_rgb(stroke)),
                    f"{line_width:.2f} w",
                    "{} B".format(" ".join(path)),
                    "Q",
                ]
            )
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        size: float = 9.0,
        bold: bool = False,
        color: str = INK,
        align: str = "left",
    ) -> None:
        width = _text_width(value, size, bold)
        if align == "center":
            x -= width / 2.0
        elif align == "right":
            x -= width
        elif align != "left":
            raise ValueError(f"Unsupported alignment: {align}")
        font = "F2" if bold else "F1"
        self.commands.append(
            "BT /{} {:.2f} Tf {:.4f} {:.4f} {:.4f} rg "
            "1 0 0 1 {:.2f} {:.2f} Tm ({}) Tj ET".format(
                font,
                size,
                *_hex_rgb(color),
                x,
                y,
                _pdf_escape(value),
            )
        )

    def to_bytes(self) -> bytes:
        content = ("\n".join(self.commands) + "\n").encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
                "/Contents 6 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            (
                f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
                + content
                + b"endstream"
            ),
        ]

        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id, body in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
            pdf.extend(body)
            pdf.extend(b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(pdf)


def draw_pdf_count_chart(
    canvas: PDFCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    summary: Sequence[Mapping[str, int]],
) -> None:
    canvas.rect(x, y, width, height, PANEL, "#DDE3E9")
    canvas.text(
        x + 16,
        y + height - 23,
        "Classification by trajectory",
        11,
        True,
    )

    plot_left = x + 48
    plot_bottom = y + 48
    plot_width = width - 68
    plot_height = 112
    max_count = 20

    for tick in (0, 5, 10, 15, 20):
        tick_x = plot_left + tick / max_count * plot_width
        canvas.line(
            tick_x,
            plot_bottom,
            tick_x,
            plot_bottom + plot_height,
            GRID,
            0.45,
        )
        canvas.text(
            tick_x,
            plot_bottom - 13,
            tick,
            6.8,
            False,
            MUTED,
            "center",
        )

    row_gap = 20
    bar_height = 11
    for index, row in enumerate(summary):
        center_y = plot_bottom + plot_height - 10 - index * row_gap
        canvas.text(
            plot_left - 10,
            center_y - 2.3,
            f"T{row['trajectory']}",
            7.5,
            True,
            INK,
            "right",
        )
        cursor = plot_left
        for key, color in (
            ("strict", STRICT_COLOR),
            ("non_strict", NON_STRICT_COLOR),
            ("unresolved", UNRESOLVED_COLOR),
        ):
            segment = row[key] / max_count * plot_width
            if segment:
                canvas.rect(
                    cursor,
                    center_y - bar_height / 2,
                    segment,
                    bar_height,
                    color,
                )
                cursor += segment
        canvas.text(
            plot_left + row["candidates"] / max_count * plot_width + 5,
            center_y - 2.3,
            row["candidates"],
            7.2,
            True,
        )

    canvas.text(
        plot_left + plot_width / 2,
        plot_bottom - 27,
        "Number of candidates",
        7.3,
        True,
        MUTED,
        "center",
    )

    legend_y = y + 13
    canvas.rect(x + 18, legend_y - 1, 9, 7, STRICT_COLOR)
    canvas.text(x + 32, legend_y, "Strict saddle", 6.8, False, MUTED)
    canvas.rect(x + 101, legend_y - 1, 9, 7, UNRESOLVED_COLOR)
    canvas.text(x + 115, legend_y, "Unresolved", 6.8, False, MUTED)
    canvas.text(
        x + width - 16,
        legend_y,
        "Non-strict = 0",
        6.8,
        True,
        NON_STRICT_COLOR,
        "right",
    )


def draw_pdf_table(
    canvas: PDFCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    summary: Sequence[Mapping[str, int]],
) -> None:
    canvas.rect(x, y, width, height, "#FFFFFF", "#DDE3E9")
    canvas.text(
        x + 16,
        y + height - 23,
        "Table 1. Numerical classification across five PROBE trajectories",
        11,
        True,
    )

    table_x = x + 18
    table_width = width - 36
    header_y = y + height - 55
    row_height = 18
    column_fractions = (0.26, 0.20, 0.18, 0.18, 0.18)
    headers = ("Trajectory", "Candidates", "Strict", "Non-strict", "Unresolved")

    starts: List[float] = []
    cursor = table_x
    for fraction in column_fractions:
        starts.append(cursor)
        cursor += table_width * fraction

    canvas.rect(
        table_x,
        header_y,
        table_width,
        row_height,
        "#EAF0F6",
    )
    for index, (header, fraction) in enumerate(
        zip(headers, column_fractions)
    ):
        center = starts[index] + table_width * fraction / 2
        canvas.text(
            center,
            header_y + 6,
            header,
            7.4,
            True,
            INK,
            "center",
        )

    rows: List[Mapping[str, int]] = list(summary) + [total_row(summary)]
    for row_index, row in enumerate(rows):
        row_y = header_y - (row_index + 1) * row_height
        is_total = row_index == len(rows) - 1
        if row_index % 2 == 1 and not is_total:
            canvas.rect(
                table_x,
                row_y,
                table_width,
                row_height,
                "#F7F9FB",
            )
        if is_total:
            canvas.line(
                table_x,
                row_y + row_height,
                table_x + table_width,
                row_y + row_height,
                INK,
                0.8,
            )
        values: Sequence[object] = (
            "Total" if is_total else row["trajectory"],
            row["candidates"],
            row["strict"],
            row["non_strict"],
            row["unresolved"],
        )
        for index, (value, fraction) in enumerate(
            zip(values, column_fractions)
        ):
            center = starts[index] + table_width * fraction / 2
            canvas.text(
                center,
                row_y + 6,
                value,
                7.4,
                is_total,
                INK,
                "center",
            )

    canvas.text(
        table_x,
        y + 17,
        "Unresolved: negative estimates were observed, but the residual "
        "bound did not certify negative curvature.",
        7.0,
        False,
        MUTED,
    )


def build_pdf(
    summary: Sequence[Mapping[str, int]],
) -> bytes:
    canvas = PDFCanvas()
    total = total_row(summary)
    conclusive = total["strict"] + total["non_strict"]

    canvas.rect(0, 0, canvas.width, canvas.height, "#FFFFFF")
    canvas.text(
        36,
        578,
        "Strict-Saddle Evidence Along Five PROBE Trajectories",
        18,
        True,
        INK,
    )
    canvas.text(
        36,
        561,
        "Perturbation-triggered lower-level stationary-point candidates",
        8.5,
        False,
        MUTED,
    )

    canvas.rect(36, 524, 720, 24, "#EAF3FA", "#BED5E8")
    canvas.rect(36, 524, 4, 24, STRICT_COLOR)
    canvas.text(
        48,
        533,
        (
            f"Results: {total['candidates']} candidates; "
            f"{total['strict']} strict; {total['non_strict']} non-strict; "
            f"{total['unresolved']} unresolved. All {conclusive} "
            "conclusively classified candidates were strict saddles."
        ),
        8.7,
        True,
        INK,
    )

    draw_pdf_count_chart(canvas, 36, 300, 720, 210, summary)
    draw_pdf_table(canvas, 36, 58, 720, 225, summary)

    canvas.text(
        36,
        35,
        "Setup: 100 outer iterations; <=200 inner steps; 5 CG steps; "
        "batch 32; LoRA rank 8; mu=0.1.",
        6.8,
        False,
        MUTED,
    )
    canvas.text(
        36,
        23,
        "Criterion: normalized gradient <=1e-3; "
        "lambda_hat_min + residual < -gamma; gamma=1e-6; "
        "two independent Lanczos runs, <=30 steps each.",
        6.8,
        False,
        MUTED,
    )
    return canvas.to_bytes()


def export_source_trajectories(
    output_dir: Path,
    csv_payload: Mapping[str, str],
) -> List[Path]:
    paths: List[Path] = []
    for name in sorted(csv_payload):
        path = output_dir / name
        # The original experiment exports use CRLF line endings.
        source_bytes = csv_payload[name].replace("\r\n", "\n")
        path.write_bytes(source_bytes.replace("\n", "\r\n").encode("utf-8"))
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate a strict-saddle PDF and tables from five "
            "embedded PROBE trajectories."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Output directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--prefix",
        default="saddle_results",
        help="Output filename prefix (default: saddle_results).",
    )
    parser.add_argument(
        "--export-trajectories",
        action="store_true",
        help="Also reconstruct the five original trajectory CSV files.",
    )
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a non-empty filename prefix.")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_payload = load_embedded_csvs()
    trajectories = parse_trajectories(csv_payload)
    records = candidate_records(trajectories)
    summary = build_summary(records)

    output_paths = [
        args.output_dir / f"{args.prefix}.pdf",
        args.output_dir / f"{args.prefix}_table.csv",
        args.output_dir / f"{args.prefix}_candidates.csv",
        args.output_dir / f"{args.prefix}.md",
    ]
    output_paths[0].write_bytes(build_pdf(summary))
    write_summary_csv(output_paths[1], summary)
    write_candidates_csv(output_paths[2], records)
    write_markdown(output_paths[3], summary)

    if args.export_trajectories:
        output_paths.extend(
            export_source_trajectories(args.output_dir, csv_payload)
        )

    total = total_row(summary)
    print(
        "Verified embedded data: "
        f"{total['candidates']} candidates, {total['strict']} strict, "
        f"{total['non_strict']} non-strict, "
        f"{total['unresolved']} unresolved."
    )
    print("Generated:")
    for path in output_paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
