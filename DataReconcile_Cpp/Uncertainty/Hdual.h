#include "define.h"
#include "fuzzy.h"

class hdual;		  	 /* расширенный класс дуальных чисел. */

class hdual
   {
   public:
      type  real;
      char  *inf_ptr;            /* битовая маска-указатель на индекс дуальной единицы.*/
      type  *inf;
      short inf_size;
      short ptr_size;            /* номер старшего единичного бита в
				    массиве inf_ptr. */
      static short   hdual::counter;	 /* счётчик дуальных единиц. */
      static fuzzy   **errors;	/* Массив с указателями на информацию о
				   погрешностях исходных данных. Каждой
				   инфинитезимальной единице соответствует
				   свой экземпляр fuzzy с ифнормацией о
				   погрешности источника. */
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

