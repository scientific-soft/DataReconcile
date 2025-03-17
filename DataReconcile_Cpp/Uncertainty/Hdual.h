#include "define.h"
#include "fuzzy.h"

class hdual;		  	 /* ���७�� ����� �㠫��� �ᥫ. */

class hdual
   {
   public:
      type  real;
      char  *inf_ptr;            /* ��⮢�� ��᪠-㪠��⥫� �� ������ �㠫쭮� �������.*/
      type  *inf;
      short inf_size;
      short ptr_size;            /* ����� ���襣� �����筮�� ��� �
				    ���ᨢ� inf_ptr. */
      static short   hdual::counter;	 /* ����稪 �㠫��� ������. */
      static fuzzy   **errors;	/* ���ᨢ � 㪠��⥫ﬨ �� ���ଠ�� �
				   ����譮���� ��室��� ������. ������
				   ��䨭�⥧����쭮� ������ ᮮ⢥�����
				   ᢮� ������� fuzzy � �䭮ଠ樥� �
				   ����譮�� ���筨��. */
   hdual(type = 0, type = 0, ErrorInfo * = NULL);
   ~hdual(void);
   hdual(hdual&);
   friend hdual operator +(hdual&, hdual&);
   friend hdual operator +(hdual&, type);
   friend hdual operator +(type, hdual&);
   friend hdual operator -(hdual&, hdual&);
   friend hdual operator -(type, hdual&);
   friend hdual operator -(hdual&, type);
   friend hdual operator /(hdual&, hdual&);
   friend hdual operator /(type, hdual&);
   friend hdual operator /(hdual&, type);
   friend hdual operator *(hdual&, hdual&);
   friend hdual operator *(type, hdual&);
   friend hdual operator *(hdual&, type);
   void  operator +=(hdual &);
   void  operator -=(hdual &);
   void  operator *=(hdual &);
   void  operator /=(hdual &);
   /*----------------------*/
   void operator =(hdual&);
   /*----------------------*/
   char  operator >=(type);
   char  operator >(type);
   char  operator <=(type);
   char  operator <(type);
   /*----------------------*/
   interval  GetErrorInterval      (type);
   fuzzy     GetErrorFuzzyVariable (void);

   /*---- Mathematical functions. -----*/
   friend hdual  acos  (hdual&);
   friend hdual  asin  (hdual&);
   friend hdual  atan  (hdual&);
   friend hdual  atan2 (hdual&, hdual&);
   friend hdual  cos   (hdual&);
   friend hdual  cosh  (hdual&);
   friend hdual  exp   (hdual&);
   friend hdual  fabs  (hdual&);
   friend hdual  log   (hdual&);
   friend hdual  log10 (hdual&);
   friend hdual  pow   (type, hdual&);
   friend hdual  pow   (hdual&, type);
   friend hdual  pow   (hdual&, hdual&);
   friend hdual  sin   (hdual&);
   friend hdual  sinh  (hdual&);
   friend hdual  sqrt  (hdual&);
   friend hdual  tan   (hdual&);
   friend hdual  tanh  (hdual&);
   };

